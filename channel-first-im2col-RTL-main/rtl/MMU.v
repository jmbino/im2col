// Explicit im2col construction, banked matrix reads, and the systolic array.
`timescale 1ns / 1ps

module MMU #(
    parameter ARRAY_SIZE = 4,
    parameter BIT_WIDTH  = 8,
    parameter ACC_WIDTH  = 32,
    parameter ADDR_WIDTH = 8,
    parameter CFG_WIDTH = 32,
    parameter IM2COL_ADDR_WIDTH = ADDR_WIDTH + 8
) (
    input [CFG_WIDTH-1:0] cfg_N,
    input [CFG_WIDTH-1:0] cfg_C,
    input [CFG_WIDTH-1:0] cfg_K,
    input [CFG_WIDTH-1:0] cfg_H,
    input [CFG_WIDTH-1:0] cfg_W,
    input [CFG_WIDTH-1:0] cfg_R,
    input [CFG_WIDTH-1:0] cfg_S,
    input [CFG_WIDTH-1:0] cfg_P,
    input [CFG_WIDTH-1:0] cfg_Q,
    input [CFG_WIDTH-1:0] cfg_pad_i,
    input [CFG_WIDTH-1:0] cfg_pad_j,
    input [CFG_WIDTH-1:0] cfg_stride_i,
    input [CFG_WIDTH-1:0] cfg_stride_j,
    input [CFG_WIDTH-1:0] cfg_dilation_i,
    input [CFG_WIDTH-1:0] cfg_dilation_j,
    input clk,
    input nrst,
    input clear_pass,
    input build_start,
    input build_active,
    output reg build_done,
    input load_weights,
    input weight_write_buffer,
    input weight_read_buffer,
    input accumulate,
    input [CFG_WIDTH-1:0] r,
    input [CFG_WIDTH-1:0] s,
    input [ARRAY_SIZE-1:0]                 sram_we_ext,
    input [ARRAY_SIZE-1:0][ADDR_WIDTH-1:0] sram_waddr_ext,
    input [ARRAY_SIZE-1:0][ BIT_WIDTH-1:0] sram_wdata_ext,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] weight_row_in,
    output wire [ARRAY_SIZE-1:0][ACC_WIDTH-1:0] acc_col_out
);

  localparam OUT_LATENCY = ARRAY_SIZE + 1;
  localparam WB_CNT_WIDTH = CFG_WIDTH + 1;
  localparam CHANNEL_INDEX_WIDTH = (ARRAY_SIZE > 1) ? $clog2(ARRAY_SIZE) : 1;

  wire [CFG_WIDTH-1:0] output_pixels = cfg_N * cfg_P * cfg_Q;
  wire [CFG_WIDTH-1:0] kernel_area = cfg_R * cfg_S;
  wire [CFG_WIDTH-1:0] reduction_terms = cfg_C * kernel_area;
  wire clear = !nrst || clear_pass;

  // The builder walks d=(c*R*S+r*S+s), then each output position m.
  // Each input-channel bank stores its R*S rows contiguously:
  // im2col[c, r*S+s, m] is at (r*S+s)*output_pixels + m.
  reg [CFG_WIDTH-1:0] build_d;
  reg [CFG_WIDTH-1:0] build_position;
  reg build_pending_q, build_pending_valid_q, build_pending_last_q;
  reg [CHANNEL_INDEX_WIDTH-1:0] build_pending_channel_q;
  reg [IM2COL_ADDR_WIDTH-1:0] build_pending_addr_q;

  wire build_request = build_active && (build_d < reduction_terms) &&
                       (build_position < output_pixels);
  wire [CFG_WIDTH-1:0] build_tap = build_d % kernel_area;
  wire [CFG_WIDTH-1:0] build_channel = build_d / kernel_area;
  wire [CFG_WIDTH-1:0] build_r = build_tap / cfg_S;
  wire [CFG_WIDTH-1:0] build_s = build_tap % cfg_S;
  wire [ADDR_WIDTH-1:0] build_source_addr;
  wire build_source_valid;
  wire [CFG_WIDTH-1:0] build_source_position =
      build_request ? build_position : {CFG_WIDTH{1'b0}};

  im2col_addr_gen #(
      .CFG_WIDTH(CFG_WIDTH), .POSITION_WIDTH(CFG_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)
  ) u_build_addr (
      .cfg_N(cfg_N), .cfg_H(cfg_H), .cfg_W(cfg_W), .cfg_Q(cfg_Q),
      .cfg_pad_i(cfg_pad_i), .cfg_pad_j(cfg_pad_j),
      .cfg_stride_i(cfg_stride_i), .cfg_stride_j(cfg_stride_j),
      .cfg_dilation_i(cfg_dilation_i), .cfg_dilation_j(cfg_dilation_j),
      .position(build_source_position), .r(build_r), .s(build_s),
      .addr(build_source_addr), .valid(build_source_valid)
  );

  wire [ADDR_WIDTH-1:0] input_read_addr = build_request
      ? build_source_addr : {ADDR_WIDTH{1'b0}};
  wire [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] raw_input_data;
  wire [ARRAY_SIZE-1:0] im2col_we;
  wire [BIT_WIDTH-1:0] im2col_wdata =
      build_pending_valid_q ? raw_input_data[build_pending_channel_q]
                            : {BIT_WIDTH{1'b0}};

  always @(posedge clk) begin
    if (!nrst || build_start) begin
      build_d <= 0;
      build_position <= 0;
      build_pending_q <= 0;
      build_pending_valid_q <= 0;
      build_pending_last_q <= 0;
      build_pending_channel_q <= 0;
      build_pending_addr_q <= 0;
      build_done <= 0;
    end else begin
      build_done <= 0;
      build_pending_q <= build_request;
      if (build_request) begin
        build_pending_valid_q <= build_source_valid;
        build_pending_last_q <= (build_d == reduction_terms-1) &&
                                (build_position == output_pixels-1);
        build_pending_channel_q <= CHANNEL_INDEX_WIDTH'(build_channel);
        build_pending_addr_q <= IM2COL_ADDR_WIDTH'(
            build_tap * output_pixels + build_position
        );
        if (build_position == output_pixels-1) begin
          build_position <= 0;
          build_d <= build_d + 1'b1;
        end else begin
          build_position <= build_position + 1'b1;
        end
      end
      // One cycle for synchronous input SRAM read and one for scratch write.
      if (build_pending_q && build_pending_last_q) build_done <= 1;
    end
  end

  genvar c;
  generate
    for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : g_build_write
      assign im2col_we[c] = build_pending_q && (build_pending_channel_q == c);
    end
  endgenerate

  wire [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] activation_rows;
  reg [CFG_WIDTH-1:0] position_cnt;
  wire [CFG_WIDTH-1:0] position_next = position_cnt + 1'b1;

  always @(posedge clk) begin
    if (clear) position_cnt <= 0;
    else position_cnt <= position_next;
  end

  wire [CFG_WIDTH-1:0] read_position =
      clear ? {CFG_WIDTH{1'b0}} : position_next;
  wire [CFG_WIDTH-1:0] tap_index = r * cfg_S + s;
  wire [IM2COL_ADDR_WIDTH-1:0] matrix_read_addr =
      IM2COL_ADDR_WIDTH'(tap_index * output_pixels + read_position);

  // The explicit matrix contains a zero for every padding coordinate, so the
  // compute stream has no per-tap padding decode.
  localparam FEED_WIDTH = IM2COL_ADDR_WIDTH + 2;
  wire [FEED_WIDTH-1:0] feed_in = {
      !clear && position_cnt < output_pixels,
      1'b1,
      matrix_read_addr
  };
  wire [ARRAY_SIZE:0][FEED_WIDTH-1:0] feed_bus;
  assign feed_bus[0] = feed_in;

  generate
    for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : g_c
      wire [ACC_WIDTH-1:0] output_channel_result;
      wire [BIT_WIDTH-1:0] input_channel_activation;

      assign output_channel_result = acc_col_out[c];

      channel_bank #(
          .CHANNEL_INDEX(c),
          .BIT_WIDTH(BIT_WIDTH),
          .ACC_WIDTH(ACC_WIDTH),
          .ADDR_WIDTH(ADDR_WIDTH),
          .IM2COL_ADDR_WIDTH(IM2COL_ADDR_WIDTH),
          .WB_CNT_WIDTH(WB_CNT_WIDTH),
          .OUT_LATENCY(OUT_LATENCY),
          .CFG_WIDTH(CFG_WIDTH)
      ) u_bank (
          .feed_active(c < cfg_C),
          .writeback_active(c < cfg_K),
          .output_pixels(output_pixels),
          .clk(clk),
          .clear(clear),
          .stream_enable(!clear_pass),
          .accumulate(accumulate),
          .feed_in(feed_bus[c]),
          .clear_feed(feed_in),
          .feed_out(feed_bus[c+1]),
          .build_input_raddr(input_read_addr),
          .build_input_rdata(raw_input_data[c]),
          .im2col_we(im2col_we[c]),
          .im2col_waddr(build_pending_addr_q),
          .im2col_wdata(im2col_wdata),
          .sram_we_ext(sram_we_ext[c]),
          .sram_waddr_ext(sram_waddr_ext[c]),
          .sram_wdata_ext(sram_wdata_ext[c]),
          .result_in(output_channel_result),
          .activation_out(input_channel_activation)
      );

      assign activation_rows[c] = input_channel_activation;
    end
  endgenerate

  systolic_array #(
      .ARRAY_SIZE(ARRAY_SIZE), .BIT_WIDTH(BIT_WIDTH),
      .ACC_WIDTH(ACC_WIDTH), .CFG_WIDTH(CFG_WIDTH)
  ) u_array (
      .clk(clk), .nrst(nrst), .clear_pass(clear_pass),
      .load_weights(load_weights), .weight_write_buffer(weight_write_buffer),
      .weight_read_buffer(weight_read_buffer), .cfg_C(cfg_C), .cfg_K(cfg_K),
      .activation_rows(activation_rows), .weight_row_in(weight_row_in),
      .acc_col_out(acc_col_out)
  );
endmodule
