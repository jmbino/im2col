// One stage of row skew, synchronous-read alignment, and activation gating.
`timescale 1ns / 1ps
module activation_feeder #(
    parameter CHANNEL_INDEX = 0, BIT_WIDTH = 8, ADDR_WIDTH = 8
) (
    input clk, clear, feed_active, stream_enable,
    input [ADDR_WIDTH+1:0] feed_in, clear_feed,
    output [ADDR_WIDTH+1:0] feed_out,
    output [ADDR_WIDTH-1:0] sram_raddr,
    input [BIT_WIDTH-1:0] sram_rdata,
    output [BIT_WIDTH-1:0] activation_out
);
  // Bundle: {feed_en, padding_valid, address}. Row zero has no delay.
  generate
    if (CHANNEL_INDEX == 0) begin : g_first
      assign feed_out = feed_in;
    end else begin : g_skew
      reg [ADDR_WIDTH+1:0] stage;
      always @(posedge clk) stage <= clear ? clear_feed : feed_in;
      assign feed_out = stage;
    end
  endgenerate

  assign sram_raddr = feed_out[ADDR_WIDTH-1:0];
  // rd_valid_q is aligned with the data returned by the synchronous SRAM.
  // Clear it between passes so a stale read cannot enter the array.
  reg rd_valid_q;
  always @(posedge clk) begin
    if (clear) rd_valid_q <= 1'b0;
    else rd_valid_q <= feed_out[ADDR_WIDTH];
  end
  assign activation_out = (feed_active && stream_enable && rd_valid_q)
                          ? sram_rdata : {BIT_WIDTH{1'b0}};
endmodule


// Separate input/output SRAM arrays, activation feeding, and output accumulation.
// Bank index selects input channel c and output channel k independently.
// The activation feeder delays the shared request by this row's position.

`timescale 1ns / 1ps

// Per-input-channel storage, explicit im2col writer, activation feed, and output accumulation.
`timescale 1ns / 1ps
module channel_bank #(
    parameter CHANNEL_INDEX = 0,
    parameter BIT_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ADDR_WIDTH = 8,
    parameter COL_ADDR_WIDTH = 16,
    parameter WB_CNT_WIDTH = 8,
    parameter OUT_LATENCY = 5,
    parameter CFG_WIDTH = 32
) (
    input feed_active,
    input writeback_active,
    input [CFG_WIDTH-1:0] N, H, W, P, Q, R, S,
    input [CFG_WIDTH-1:0] pad_i, pad_j, stride_i, stride_j, dilation_i, dilation_j,
    input im2col_start,
    output im2col_done,
    input [CFG_WIDTH-1:0] output_pixels,
    input clk, nrst, clear, stream_enable, accumulate,
    input [COL_ADDR_WIDTH+1:0] feed_in, clear_feed,
    output [COL_ADDR_WIDTH+1:0] feed_out,
    input sram_we_ext,
    input [ADDR_WIDTH-1:0] sram_waddr_ext,
    input [BIT_WIDTH-1:0] sram_wdata_ext,
    input [ACC_WIDTH-1:0] result_in,
    output [BIT_WIDTH-1:0] activation_out
);
  wire [BIT_WIDTH-1:0] input_rdata, col_rdata;
  wire [ACC_WIDTH-1:0] old_out, wb_data;
  wire [ADDR_WIDTH-1:0] wb_ptr, wb_ptr_next;
  wire wb_valid;
  wire [ADDR_WIDTH-1:0] wb_unused_addr;
  wire [COL_ADDR_WIDTH-1:0] col_raddr;
  wire [ADDR_WIDTH-1:0] input_raddr;
  wire col_we;
  wire [COL_ADDR_WIDTH-1:0] col_waddr;
  wire [BIT_WIDTH-1:0] col_wdata;
  wire writer_busy, writer_done;

  // Original feature-map SRAM is written externally and read only during transform.
  row_sram #(.DATA_WIDTH(BIT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) u_input_sram (
      .clk(clk), .we(sram_we_ext), .waddr(sram_waddr_ext), .wdata(sram_wdata_ext),
      .raddr(input_raddr), .rdata(input_rdata)
  );

  explicit_im2col_writer #(
      .CFG_WIDTH(CFG_WIDTH), .INPUT_ADDR_WIDTH(ADDR_WIDTH),
      .COL_ADDR_WIDTH(COL_ADDR_WIDTH), .DATA_WIDTH(BIT_WIDTH)
  ) u_im2col_writer (
      .clk(clk), .nrst(nrst), .start(im2col_start && feed_active),
      .N(N), .H(H), .W(W), .P(P), .Q(Q), .R(R), .S(S),
      .pad_i(pad_i), .pad_j(pad_j), .stride_i(stride_i), .stride_j(stride_j),
      .dilation_i(dilation_i), .dilation_j(dilation_j),
      .input_raddr(input_raddr), .input_rdata(input_rdata),
      .col_we(col_we), .col_waddr(col_waddr), .col_wdata(col_wdata),
      .busy(writer_busy), .done(writer_done)
  );
  assign im2col_done = !feed_active || writer_done;

  // Materialized matrix is read during compute. Tap/position addressing is supplied by MMU.
  row_sram #(.DATA_WIDTH(BIT_WIDTH), .ADDR_WIDTH(COL_ADDR_WIDTH)) u_col_sram (
      .clk(clk), .we(col_we), .waddr(col_waddr), .wdata(col_wdata),
      .raddr(col_raddr), .rdata(col_rdata)
  );

  activation_feeder #(
      .CHANNEL_INDEX(CHANNEL_INDEX), .BIT_WIDTH(BIT_WIDTH), .ADDR_WIDTH(COL_ADDR_WIDTH)
  ) u_activation (
      .clk(clk), .clear(clear), .feed_active(feed_active), .stream_enable(stream_enable),
      .feed_in(feed_in), .clear_feed(clear_feed), .feed_out(feed_out),
      .sram_raddr(col_raddr), .sram_rdata(col_rdata), .activation_out(activation_out)
  );

  output_accumulator #(
      .CHANNEL_INDEX(CHANNEL_INDEX), .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH),
      .WB_CNT_WIDTH(WB_CNT_WIDTH), .OUT_LATENCY(OUT_LATENCY), .CFG_WIDTH(CFG_WIDTH)
  ) u_output (
      .clk(clk), .clear(clear), .writeback_active(writeback_active),
      .stream_enable(stream_enable), .accumulate(accumulate), .output_pixels(output_pixels),
      .old_out(old_out), .result_in(result_in), .wb_valid(wb_valid),
      .wb_ptr(wb_ptr), .wb_ptr_next(wb_ptr_next), .wb_data(wb_data)
  );

  row_sram #(.DATA_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) u_output_sram (
      .clk(clk), .we(wb_valid), .waddr(wb_ptr), .wdata(wb_data),
      .raddr(wb_ptr_next), .rdata(old_out)
  );
endmodule

// Tap sequencing, pass duration, and active PE weight-buffer selection.
`timescale 1ns / 1ps
// Controls explicit im2col preprocessing followed by tap sequencing and systolic execution.
`timescale 1ns / 1ps
module conv_controller #(parameter ARRAY_SIZE = 4, CFG_WIDTH = 32) (
    input clk, nrst, start, initial_done, im2col_done,
    input [CFG_WIDTH-1:0] N, P, Q, R, S,
    output busy,
    output reg done,
    output im2col_start,
    output clear_pass, prime, initial_load, running, preload_next, restart_weights, accumulate,
    output reg active_weight_buffer,
    output reg [CFG_WIDTH-1:0] r, s,
    output [CFG_WIDTH-1:0] next_r, next_s
);
  localparam IDLE=3'd0, IM2COL_START=3'd1, WAIT_IM2COL=3'd2,
             PRIME=3'd3, LOAD_WEIGHTS=3'd4, RUN=3'd5, SWITCH_TAP=3'd6;
  reg [2:0] state;
  reg [CFG_WIDTH:0] run_count;
  wire [CFG_WIDTH:0] run_cycles = (CFG_WIDTH+1)'(N) * P * Q + 2*ARRAY_SIZE;
  assign busy = state != IDLE;
  assign im2col_start = state == IM2COL_START;
  assign clear_pass = !nrst || state != RUN;
  wire has_next_tap = r != R-1 || s != S-1;
  assign next_r = s == S-1 ? r + 1'b1 : r;
  assign next_s = s == S-1 ? {CFG_WIDTH{1'b0}} : s + 1'b1;
  assign preload_next = state == RUN && has_next_tap;
  assign prime = state == PRIME;
  assign initial_load = state == LOAD_WEIGHTS;
  assign running = state == RUN;
  assign accumulate = r != 0 || s != 0;
  assign restart_weights = (state == IDLE && start) ||
                           (state == RUN && run_count == run_cycles-1 && has_next_tap);

  always @(posedge clk) begin
    if (!nrst) begin
      state <= IDLE; done <= 0; active_weight_buffer <= 0;
      r <= 0; s <= 0; run_count <= 0;
    end else begin
      done <= 0;
      case (state)
        IDLE: if (start) begin
          active_weight_buffer <= 0; r <= 0; s <= 0; run_count <= 0;
          state <= IM2COL_START;
        end
        IM2COL_START: state <= WAIT_IM2COL;
        WAIT_IM2COL: if (im2col_done) state <= PRIME;
        PRIME: state <= LOAD_WEIGHTS;
        LOAD_WEIGHTS: if (initial_done) begin run_count <= 0; state <= RUN; end
        SWITCH_TAP: begin run_count <= 0; state <= RUN; end
        RUN: begin
          if (run_count == run_cycles-1) begin
            run_count <= 0;
            if (s == S-1 && r == R-1) begin state <= IDLE; done <= 1; end
            else begin
              r <= next_r; s <= next_s; active_weight_buffer <= !active_weight_buffer;
              state <= SWITCH_TAP;
            end
          end else run_count <= run_count + 1'b1;
        end
        default: state <= IDLE;
      endcase
    end
  end
endmodule

`timescale 1ns / 1ps

// Runtime configuration capture and wiring for the convolution controller and datapath.
// Images share weights and stream in spatial-position-major, image-minor order.
// Configuration is captured on start and remains fixed throughout the run.
// Captures configuration, materializes im2col, then runs the existing systolic datapath.
`timescale 1ns / 1ps
module conv_core #(
    parameter ARRAY_SIZE = 4, BIT_WIDTH = 8, ACC_WIDTH = 32,
    parameter ADDR_WIDTH = 8, KERNEL_ADDR_WIDTH = 8,
    parameter COL_ADDR_WIDTH = 16, CFG_WIDTH = 32
) (
    input clk, nrst, start,
    output busy, done,
    input [CFG_WIDTH-1:0] cfg_N, cfg_C, cfg_H, cfg_W, cfg_K, cfg_R, cfg_S, cfg_P, cfg_Q,
    input [CFG_WIDTH-1:0] cfg_pad_i, cfg_pad_j, cfg_stride_i, cfg_stride_j,
    input [CFG_WIDTH-1:0] cfg_dilation_i, cfg_dilation_j,
    input [ARRAY_SIZE-1:0] sram_we_ext,
    input [ARRAY_SIZE-1:0][ADDR_WIDTH-1:0] sram_waddr_ext,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] sram_wdata_ext,
    input [ARRAY_SIZE-1:0] kernel_we_ext,
    input [ARRAY_SIZE-1:0][KERNEL_ADDR_WIDTH-1:0] kernel_waddr_ext,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] kernel_wdata_ext,
    output [ARRAY_SIZE-1:0][ACC_WIDTH-1:0] acc_col_out
);
  reg [CFG_WIDTH-1:0] N,C,H,W,K,R,S,P,Q,pad_i,pad_j,stride_i,stride_j,dilation_i,dilation_j;
  wire [CFG_WIDTH-1:0] r,s,next_r,next_s;
  wire clear_pass,prime,initial_load,running,preload_next,restart_weights,accumulate;
  wire active_weight_buffer,im2col_start,im2col_done;
  wire load_weights,weight_write_buffer;
  wire [ARRAY_SIZE-1:0] input_we = sram_we_ext & {ARRAY_SIZE{nrst && !busy}};
  wire initial_done,kernel_read_enable;
  wire [CFG_WIDTH-1:0] c_read;
  wire [KERNEL_ADDR_WIDTH-1:0] kernel_raddr;
  wire [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] kernel_rdata,weight_row_in;

  weight_loader #(.ARRAY_SIZE(ARRAY_SIZE),.BIT_WIDTH(BIT_WIDTH),.ADDR_WIDTH(KERNEL_ADDR_WIDTH),.CFG_WIDTH(CFG_WIDTH)) u_weight_loader (
    .clk(clk),.nrst(nrst),.restart(restart_weights),.prime(prime),.initial_load(initial_load),
    .preload_next(preload_next),.running(running),.active_weight_buffer(active_weight_buffer),
    .R(R),.S(S),.r(r),.s(s),.next_r(next_r),.next_s(next_s),.kernel_rdata(kernel_rdata),
    .load_weights(load_weights),.weight_write_buffer(weight_write_buffer),.initial_done(initial_done),
    .kernel_read_enable(kernel_read_enable),.kernel_raddr(kernel_raddr),.c_read(c_read),.weight_row_in(weight_row_in)
  );

  genvar k;
  generate for (k=0;k<ARRAY_SIZE;k=k+1) begin : g_kernel
    kernel_bank #(.BIT_WIDTH(BIT_WIDTH),.ADDR_WIDTH(KERNEL_ADDR_WIDTH)) u_bank (
      .clk(clk),.we(nrst && !busy && kernel_we_ext[k]),.waddr(kernel_waddr_ext[k]),.wdata(kernel_wdata_ext[k]),
      .read_enable(nrst && kernel_read_enable),.read_active(k<K && c_read<C),
      .raddr(kernel_raddr),.rdata(kernel_rdata[k])
    );
  end endgenerate

  always @(posedge clk) begin
    if (!nrst) begin
      N<=1; C<=0; H<=0; W<=0; K<=0; R<=0; S<=0; P<=0; Q<=0;
      pad_i<=0; pad_j<=0; stride_i<=0; stride_j<=0; dilation_i<=0; dilation_j<=0;
    end else if (start && !busy) begin
      N<=cfg_N; C<=cfg_C; H<=cfg_H; W<=cfg_W; K<=cfg_K; R<=cfg_R; S<=cfg_S; P<=cfg_P; Q<=cfg_Q;
      pad_i<=cfg_pad_i; pad_j<=cfg_pad_j; stride_i<=cfg_stride_i; stride_j<=cfg_stride_j;
      dilation_i<=cfg_dilation_i; dilation_j<=cfg_dilation_j;
    end
  end

  conv_controller #(.ARRAY_SIZE(ARRAY_SIZE),.CFG_WIDTH(CFG_WIDTH)) u_controller (
    .clk(clk),.nrst(nrst),.start(start),.initial_done(initial_done),.im2col_done(im2col_done),
    .N(N),.P(P),.Q(Q),.R(R),.S(S),.busy(busy),.done(done),.im2col_start(im2col_start),
    .clear_pass(clear_pass),.prime(prime),.initial_load(initial_load),.running(running),
    .preload_next(preload_next),.restart_weights(restart_weights),.accumulate(accumulate),
    .active_weight_buffer(active_weight_buffer),.r(r),.s(s),.next_r(next_r),.next_s(next_s)
  );

  MMU #(.ARRAY_SIZE(ARRAY_SIZE),.BIT_WIDTH(BIT_WIDTH),.ACC_WIDTH(ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),.COL_ADDR_WIDTH(COL_ADDR_WIDTH),.CFG_WIDTH(CFG_WIDTH)) u_mmu (
    .cfg_N(N),.cfg_C(C),.cfg_K(K),.cfg_H(H),.cfg_W(W),.cfg_P(P),.cfg_Q(Q),.cfg_R(R),.cfg_S(S),
    .cfg_pad_i(pad_i),.cfg_pad_j(pad_j),.cfg_stride_i(stride_i),.cfg_stride_j(stride_j),
    .cfg_dilation_i(dilation_i),.cfg_dilation_j(dilation_j),.im2col_start(im2col_start),.im2col_done(im2col_done),
    .r(r),.s(s),.clk(clk),.nrst(nrst),.clear_pass(clear_pass),.load_weights(load_weights),
    .weight_write_buffer(weight_write_buffer),.weight_read_buffer(active_weight_buffer),.accumulate(accumulate),
    .sram_we_ext(input_we),.sram_waddr_ext(sram_waddr_ext),.sram_wdata_ext(sram_wdata_ext),
    .weight_row_in(weight_row_in),.acc_col_out(acc_col_out)
  );
endmodule

`timescale 1ns / 1ps

// Combinational input SRAM address generation for a convolution tap.
// Shapes: input (N,C,H,W), kernel (K,C,R,S), output (N,K,P,Q).
// position = (p*Q+q)*N+n: images vary fastest within each output position.
// addr = (h*W+w)*N+n within the SRAM array for input channel c.
module im2col_addr_gen #(
    parameter CFG_WIDTH = 32,
    parameter POSITION_WIDTH  = 8,
    parameter ADDR_WIDTH = 8
) (
    input [CFG_WIDTH-1:0] cfg_N,
    input [CFG_WIDTH-1:0] cfg_H,
    input [CFG_WIDTH-1:0] cfg_W,
    input [CFG_WIDTH-1:0] cfg_Q,
    input [CFG_WIDTH-1:0] cfg_pad_i,
    input [CFG_WIDTH-1:0] cfg_pad_j,
    input [CFG_WIDTH-1:0] cfg_stride_i,
    input [CFG_WIDTH-1:0] cfg_stride_j,
    input [CFG_WIDTH-1:0] cfg_dilation_i,
    input [CFG_WIDTH-1:0] cfg_dilation_j,
    input  [POSITION_WIDTH-1:0] position,
    input  [CFG_WIDTH-1:0]  r,
    input  [CFG_WIDTH-1:0]  s,
    output [ADDR_WIDTH-1:0] addr,
    output                 valid
);

  // Decode image and spatial coordinates from the interleaved stream.
  wire [POSITION_WIDTH-1:0] n = position % cfg_N;
  wire [POSITION_WIDTH-1:0] pix = position / cfg_N;
  wire [POSITION_WIDTH-1:0] p = pix / cfg_Q;
  wire [POSITION_WIDTH-1:0] q = pix % cfg_Q;

  // Zero-extend unsigned coordinates before signed arithmetic. Negative h/w
  // represent padding above/left of the image. Wide intermediates preserve
  // the products for configurable pixel widths and convolution geometry.
  wire signed [63:0] p_signed = $signed({1'b0, p});
  wire signed [63:0] q_signed = $signed({1'b0, q});
  wire signed [63:0] r_signed = $signed({1'b0, r});
  wire signed [63:0] s_signed = $signed({1'b0, s});
  wire signed [63:0] h = p_signed * cfg_stride_i + r_signed * cfg_dilation_i - cfg_pad_i;
  wire signed [63:0] w = q_signed * cfg_stride_j + s_signed * cfg_dilation_j - cfg_pad_j;

  // The caller bounds position and tap coordinates; padding checks use h/w.
  assign valid = (h >= 0) && (h < cfg_H) && (w >= 0) && (w < cfg_W);
  wire signed [63:0] flat = h * cfg_W + w;
  assign addr = valid ? ADDR_WIDTH'(flat * cfg_N + n) : {ADDR_WIDTH{1'b0}};
endmodule




// One output channel's kernel memory with synchronous reads.
`timescale 1ns / 1ps
module kernel_bank #(parameter BIT_WIDTH = 8, ADDR_WIDTH = 8) (
    input clk, we, read_enable, read_active,
    input [ADDR_WIDTH-1:0] waddr, raddr,
    input [BIT_WIDTH-1:0] wdata,
    output reg [BIT_WIDTH-1:0] rdata
);
  reg [BIT_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    if (read_enable) rdata <= read_active ? mem[raddr] : {BIT_WIDTH{1'b0}};
  end
endmodule


`timescale 1ns / 1ps

// Signed PE with two stationary weights: compute from one while loading the other.
module MAC #(
    parameter BIT_WIDTH = 8,
    parameter ACC_WIDTH = 32
) (
    input clk,
    input nrst,
    input clear_acc,
    input load_weights,
    input weight_write_buffer,
    input weight_read_buffer,
    input signed [ACC_WIDTH-1:0] acc_in,
    input signed [BIT_WIDTH-1:0] data_in,
    input signed [BIT_WIDTH-1:0] weight_in,
    output reg signed [ACC_WIDTH-1:0] acc_out,
    output reg signed [BIT_WIDTH-1:0] data_out,
    output reg signed [BIT_WIDTH-1:0] weight_out
);
  reg signed [BIT_WIDTH-1:0] stored_weight [0:1];
  wire signed [ACC_WIDTH-1:0] product =
      $signed(data_in) * $signed(stored_weight[weight_read_buffer]);

  always @(posedge clk) begin
    if (!nrst) begin
      acc_out <= 0;
      data_out <= 0;
      weight_out <= 0;
      stored_weight[0] <= 0;
      stored_weight[1] <= 0;
    end else begin
      // Weight shifting is independent of activation and partial-sum advancement.
      if (load_weights) begin
        stored_weight[weight_write_buffer] <= weight_in;
        weight_out <= weight_in;
      end
      if (clear_acc) begin
        acc_out <= 0;
        data_out <= 0;
      end else begin
        data_out <= data_in;
        acc_out <= acc_in + product;
      end
    end
  end
endmodule




// Square systolic array for one convolution tap, with skewed input reads and SRAM writeback.

`timescale 1ns / 1ps

// MMU reads the precomputed im2col matrix; no spatial address generation occurs during compute.
`timescale 1ns / 1ps
module MMU #(
    parameter ARRAY_SIZE=4, BIT_WIDTH=8, ACC_WIDTH=32,
    parameter ADDR_WIDTH=8, COL_ADDR_WIDTH=16, CFG_WIDTH=32
) (
    input [CFG_WIDTH-1:0] cfg_N,cfg_C,cfg_K,cfg_H,cfg_W,cfg_P,cfg_Q,cfg_R,cfg_S,
    input [CFG_WIDTH-1:0] cfg_pad_i,cfg_pad_j,cfg_stride_i,cfg_stride_j,cfg_dilation_i,cfg_dilation_j,
    input im2col_start,
    output im2col_done,
    input [CFG_WIDTH-1:0] r,s,
    input clk,nrst,clear_pass,load_weights,weight_write_buffer,weight_read_buffer,accumulate,
    input [ARRAY_SIZE-1:0] sram_we_ext,
    input [ARRAY_SIZE-1:0][ADDR_WIDTH-1:0] sram_waddr_ext,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] sram_wdata_ext,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] weight_row_in,
    output wire [ARRAY_SIZE-1:0][ACC_WIDTH-1:0] acc_col_out
);
  localparam OUT_LATENCY=ARRAY_SIZE+1;
  localparam WB_CNT_WIDTH=CFG_WIDTH+1;
  wire [CFG_WIDTH-1:0] output_pixels=cfg_N*cfg_P*cfg_Q;
  wire clear=!nrst || clear_pass;
  wire [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] activation_rows;
  reg [CFG_WIDTH-1:0] position_cnt;
  always @(posedge clk) begin
    if (clear) position_cnt <= {CFG_WIDTH{1'b0}};
    else if (position_cnt < output_pixels) position_cnt <= position_cnt + 1'b1;
  end

  // Request positions 0 through output_pixels-1 in tap-major order.
  // Clamp the address during bubbles and flush cycles to keep SRAM accesses in range.
  wire feed_valid = !clear && (position_cnt < output_pixels);
  wire [CFG_WIDTH-1:0] read_position = position_cnt;
  wire [COL_ADDR_WIDTH-1:0] col_addr = feed_valid
      ? COL_ADDR_WIDTH'((r*cfg_S+s)*output_pixels + read_position)
      : {COL_ADDR_WIDTH{1'b0}};
  localparam FW=COL_ADDR_WIDTH+2;
  wire [FW-1:0] feed_in = {feed_valid, feed_valid, col_addr};
  wire [ARRAY_SIZE:0][FW-1:0] feed_bus;
  assign feed_bus[0]=feed_in;

  wire [ARRAY_SIZE-1:0] channel_im2col_done;
  genvar c;
  generate for(c=0;c<ARRAY_SIZE;c=c+1) begin : g_c
    wire [ACC_WIDTH-1:0] output_channel_result=acc_col_out[c];
    wire [BIT_WIDTH-1:0] input_channel_activation;
    channel_bank #(.CHANNEL_INDEX(c),.BIT_WIDTH(BIT_WIDTH),.ACC_WIDTH(ACC_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH),.COL_ADDR_WIDTH(COL_ADDR_WIDTH),.WB_CNT_WIDTH(WB_CNT_WIDTH),
      .OUT_LATENCY(OUT_LATENCY),.CFG_WIDTH(CFG_WIDTH)) u_bank (
      .feed_active(c<cfg_C),.writeback_active(c<cfg_K),
      .N(cfg_N),.H(cfg_H),.W(cfg_W),.P(cfg_P),.Q(cfg_Q),.R(cfg_R),.S(cfg_S),
      .pad_i(cfg_pad_i),.pad_j(cfg_pad_j),.stride_i(cfg_stride_i),.stride_j(cfg_stride_j),
      .dilation_i(cfg_dilation_i),.dilation_j(cfg_dilation_j),
      .im2col_start(im2col_start),.im2col_done(channel_im2col_done[c]),.output_pixels(output_pixels),
      .clk(clk),.nrst(nrst),.clear(clear),.stream_enable(!clear_pass),.accumulate(accumulate),
      .feed_in(feed_bus[c]),.clear_feed(feed_in),.feed_out(feed_bus[c+1]),
      .sram_we_ext(sram_we_ext[c]),.sram_waddr_ext(sram_waddr_ext[c]),.sram_wdata_ext(sram_wdata_ext[c]),
      .result_in(output_channel_result),.activation_out(input_channel_activation)
    );
    assign activation_rows[c]=input_channel_activation;
  end endgenerate
  assign im2col_done = &channel_im2col_done;

  systolic_array #(.ARRAY_SIZE(ARRAY_SIZE),.BIT_WIDTH(BIT_WIDTH),.ACC_WIDTH(ACC_WIDTH),.CFG_WIDTH(CFG_WIDTH)) u_array (
    .clk(clk),.nrst(nrst),.clear_pass(clear_pass),.load_weights(load_weights),
    .weight_write_buffer(weight_write_buffer),.weight_read_buffer(weight_read_buffer),
    .cfg_C(cfg_C),.cfg_K(cfg_K),.activation_rows(activation_rows),.weight_row_in(weight_row_in),.acc_col_out(acc_col_out)
  );
endmodule

// Address and data generation for per-tap output writeback.
`timescale 1ns / 1ps
module output_accumulator #(
    parameter CHANNEL_INDEX = 0, ACC_WIDTH = 32, ADDR_WIDTH = 8,
              WB_CNT_WIDTH = 8, OUT_LATENCY = 5, CFG_WIDTH = 32
) (
    input clk, clear, writeback_active, stream_enable, accumulate,
    input [CFG_WIDTH-1:0] output_pixels,
    input [ACC_WIDTH-1:0] old_out, result_in,
    output wb_valid,
    output reg [ADDR_WIDTH-1:0] wb_ptr,
    output [ADDR_WIDTH-1:0] wb_ptr_next,
    output [ACC_WIDTH-1:0] wb_data
);
  // The registered column result arrives one cycle after the nominal array
  // latency. Delay writeback with it so result zero is stored at address zero.
  localparam WB_START_CYCLE = CHANNEL_INDEX + OUT_LATENCY + 1;
  reg [WB_CNT_WIDTH-1:0] writeback_cnt;
  always @(posedge clk) begin
    if (clear) writeback_cnt <= 0;
    else writeback_cnt <= writeback_cnt + 1'b1;
  end

  // The output SRAM starts at address zero.
  assign wb_valid = writeback_active && stream_enable && (writeback_cnt >= WB_START_CYCLE) &&
                  (writeback_cnt - WB_START_CYCLE < output_pixels);

  // Advance the output address on each enabled writeback cycle.
  always @(posedge clk) begin
    if (clear) wb_ptr <= 0;
    else if (wb_valid) wb_ptr <= wb_ptr + 1'b1;
  end

  // Across-taps accumulation: read-modify-write on the output plane.
  // The SRAM has one-cycle read latency, so we read the address wb_ptr will have on the next cycle;
  // the old value then shows up on old_out in the same cycle result_in is written.
  // Each output address is written once per pass and read the cycle before, so there is
  // no read-after-write hazard within a pass.
  assign wb_ptr_next = clear ? {ADDR_WIDTH{1'b0}} : (wb_valid ? wb_ptr + 1'b1 : wb_ptr);

  assign wb_data = accumulate ? old_out + result_in : result_in;
endmodule



// SRAM model with one write port and one synchronous read port.


`timescale 1ns / 1ps

module row_sram #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 8
) (
    input                  clk,
    input                  we,
    input [ADDR_WIDTH-1:0] waddr,
    input [ DATA_WIDTH-1:0] wdata,
    input [ADDR_WIDTH-1:0] raddr,

    output reg [DATA_WIDTH-1:0] rdata
);
  localparam DEPTH = 1 << ADDR_WIDTH;

  reg [DATA_WIDTH-1:0] mem[0:DEPTH-1];

  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end

endmodule



// PE grid, stationary weight buffers, and registered column outputs.
`timescale 1ns / 1ps
module systolic_array #(
    parameter ARRAY_SIZE = 4, BIT_WIDTH = 8, ACC_WIDTH = 32, CFG_WIDTH = 32
) (
    input clk, nrst, clear_pass, load_weights,
    input weight_write_buffer, weight_read_buffer,
    input [CFG_WIDTH-1:0] cfg_C, cfg_K,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] activation_rows, weight_row_in,
    output reg [ARRAY_SIZE-1:0][ACC_WIDTH-1:0] acc_col_out
);
  // Each PE connection is a separate word-sized net.
  wire [BIT_WIDTH-1:0] mac_data_in [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1], mac_data_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];     // Buses between columns
  wire [BIT_WIDTH-1:0] mac_weight_in [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1], mac_weight_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]; // Buses between rows
  wire [ACC_WIDTH-1:0] mac_acc_in [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1], mac_acc_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];       // Buses between rows

  genvar c, k;
  generate
    for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : g_c
      for (k = 0; k < ARRAY_SIZE; k = k + 1) begin : g_k

        // Column k=0 receives the SRAM activation; later columns receive it from the left PE.
        if (k == 0) begin
          assign mac_data_in[c][k] = activation_rows[c];
        end else begin
          assign mac_data_in[c][k] = mac_data_out[c][k-1];
        end

        // Input channel c=0 starts the sum; later channels add the preceding PE result.
        if (c == 0) begin
          assign mac_acc_in[c][k]    = 0;
          assign mac_weight_in[c][k] = (load_weights && k < cfg_K) ? weight_row_in[k] : 0;
        end else begin
          assign mac_acc_in[c][k]    = mac_acc_out[c-1][k];
          assign mac_weight_in[c][k] = (load_weights && c < cfg_C && k < cfg_K) ? mac_weight_out[c-1][k] : 0;
        end

        MAC #(
            .BIT_WIDTH(BIT_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) u_MAC (
            .nrst        (nrst),
            .clk         (clk),
            .clear_acc(clear_pass),
            .load_weights(load_weights),
            .weight_write_buffer(weight_write_buffer),
            .weight_read_buffer(weight_read_buffer),
            .data_in     (mac_data_in[c][k]),
            .weight_in   (mac_weight_in[c][k]),
            .acc_in      (mac_acc_in[c][k]),
            .data_out    (mac_data_out[c][k]),
            .weight_out  (mac_weight_out[c][k]),
            .acc_out     (mac_acc_out[c][k])
        );

      end
    end
  endgenerate

  generate
    for (k = 0; k < ARRAY_SIZE; k = k + 1) begin : g_output
      always @(posedge clk) begin
        if (!nrst) acc_col_out[k] <= 0;
        else acc_col_out[k] <= mac_acc_out[ARRAY_SIZE-1][k];
      end
    end
  endgenerate

endmodule



// Prefetch and descending row shifts for the initial or next tap.
`timescale 1ns / 1ps
module weight_loader #(
    parameter ARRAY_SIZE = 4, BIT_WIDTH = 8, ADDR_WIDTH = 8, CFG_WIDTH = 32
) (
    input clk, nrst, restart, prime, initial_load, preload_next, running,
    input active_weight_buffer,
    input [CFG_WIDTH-1:0] R, S, r, s, next_r, next_s,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] kernel_rdata,
    output load_weights, weight_write_buffer, initial_done, kernel_read_enable,
    output [ADDR_WIDTH-1:0] kernel_raddr,
    output [CFG_WIDTH-1:0] c_read,
    output [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] weight_row_in
);
  localparam C_WIDTH = (ARRAY_SIZE > 1) ? $clog2(ARRAY_SIZE) : 1;
  localparam PREFETCH = 2'd0, SHIFT_WEIGHTS = 2'd1, WEIGHTS_READY = 2'd2;
  reg [1:0] weight_phase;
  reg [C_WIDTH-1:0] c_load;
  assign load_weights = initial_load || (preload_next && weight_phase == SHIFT_WEIGHTS);
  assign weight_write_buffer = running ? !active_weight_buffer : active_weight_buffer;
  assign initial_done = c_load == 0;
  wire [C_WIDTH-1:0] read_row = load_weights && c_load != 0 ? c_load - 1'b1 : c_load;
  assign c_read = CFG_WIDTH'(read_row);
  wire [CFG_WIDTH-1:0] kernel_r = running ? next_r : r;
  wire [CFG_WIDTH-1:0] kernel_s = running ? next_s : s;
  assign kernel_read_enable = prime || initial_load || (preload_next && weight_phase != WEIGHTS_READY);
  assign kernel_raddr = ADDR_WIDTH'((c_read*R + kernel_r)*S + kernel_s);
  assign weight_row_in = load_weights ? kernel_rdata : {ARRAY_SIZE*BIT_WIDTH{1'b0}};
  always @(posedge clk) begin
    if (!nrst || restart) begin
      c_load <= C_WIDTH'(ARRAY_SIZE-1);
      weight_phase <= PREFETCH;
    end else if (initial_load) begin
      if (initial_done) begin
        c_load <= C_WIDTH'(ARRAY_SIZE-1);
        weight_phase <= PREFETCH;
      end else c_load <= c_load - 1'b1;
    end else if (preload_next) begin
      case (weight_phase)
        PREFETCH: weight_phase <= SHIFT_WEIGHTS;
        SHIFT_WEIGHTS: begin
          if (c_load == 0) weight_phase <= WEIGHTS_READY;
          else c_load <= c_load - 1'b1;
        end
        WEIGHTS_READY: begin end
      endcase
    end
  end
endmodule































































`timescale 1ns / 1ps

// Explicit im2col writer for ONE input channel.
//
// Input SRAM layout:
//   input_addr = ((h * W) + w) * N + n
//
// Output im2col SRAM layout:
//   output_addr = ((r * S + s) * (N*P*Q)) + position
//   position    = (p * Q + q) * N + n
//
// The input SRAM has a synchronous read port.
// The controller issues one read, then writes the returned value
// (or zero for padding) to the im2col SRAM.
//
// This module generates write signals; instantiate a separate
// output SRAM for each channel, or use a suitably banked SRAM.

// Explicit im2col writer for one input channel, with independent source and destination address widths.
`timescale 1ns / 1ps
module explicit_im2col_writer #(
    parameter CFG_WIDTH=32, INPUT_ADDR_WIDTH=8, COL_ADDR_WIDTH=16, DATA_WIDTH=8
) (
    input clk,nrst,start,
    input [CFG_WIDTH-1:0] N,H,W,P,Q,R,S,
    input [CFG_WIDTH-1:0] pad_i,pad_j,stride_i,stride_j,dilation_i,dilation_j,
    output [INPUT_ADDR_WIDTH-1:0] input_raddr,
    input [DATA_WIDTH-1:0] input_rdata,
    output col_we,
    output [COL_ADDR_WIDTH-1:0] col_waddr,
    output [DATA_WIDTH-1:0] col_wdata,
    output busy,
    output reg done
);
  localparam IDLE=2'd0, ISSUE=2'd1, WRITE=2'd2, FINISH=2'd3;
  reg [1:0] state;
  reg [CFG_WIDTH-1:0] position,tap_r,tap_s;
  reg [COL_ADDR_WIDTH-1:0] saved_col_addr;
  reg saved_valid;
  wire [CFG_WIDTH-1:0] output_pixels=N*P*Q;
  wire [CFG_WIDTH-1:0] n_idx=(N!=0) ? position%N : 0;
  wire [CFG_WIDTH-1:0] pix_idx=(N!=0) ? position/N : 0;
  wire [CFG_WIDTH-1:0] p_idx=(Q!=0) ? pix_idx/Q : 0;
  wire [CFG_WIDTH-1:0] q_idx=(Q!=0) ? pix_idx%Q : 0;
  wire signed [63:0] h_coord=$signed({1'b0,p_idx})*$signed({1'b0,stride_i})+
      $signed({1'b0,tap_r})*$signed({1'b0,dilation_i})-$signed({1'b0,pad_i});
  wire signed [63:0] w_coord=$signed({1'b0,q_idx})*$signed({1'b0,stride_j})+
      $signed({1'b0,tap_s})*$signed({1'b0,dilation_j})-$signed({1'b0,pad_j});
  wire access_valid=(N!=0)&&(Q!=0)&&(h_coord>=0)&&(h_coord<$signed({1'b0,H}))&&
      (w_coord>=0)&&(w_coord<$signed({1'b0,W}));
  wire [63:0] input_addr_calc=(h_coord*$signed({1'b0,W})+w_coord)*$signed({1'b0,N})+n_idx;
  wire [63:0] col_addr_calc=(tap_r*S+tap_s)*output_pixels+position;
  assign input_raddr=(state==ISSUE && access_valid) ? INPUT_ADDR_WIDTH'(input_addr_calc) : {INPUT_ADDR_WIDTH{1'b0}};
  assign col_we=nrst && (state==WRITE);
  assign col_waddr=saved_col_addr;
  assign col_wdata=saved_valid ? input_rdata : {DATA_WIDTH{1'b0}};
  assign busy=(state!=IDLE)&&(state!=FINISH);

  always @(posedge clk) begin
    if(!nrst) begin
      state<=IDLE; position<=0; tap_r<=0; tap_s<=0; saved_col_addr<=0; saved_valid<=0; done<=0;
    end else begin
      done<=0;
      case(state)
        IDLE: if(start && N!=0 && P!=0 && Q!=0 && R!=0 && S!=0) begin
          position<=0; tap_r<=0; tap_s<=0; state<=ISSUE;
        end
        ISSUE: begin
          // Synchronous input SRAM samples input_raddr on this edge.
          saved_col_addr<=COL_ADDR_WIDTH'(col_addr_calc);
          saved_valid<=access_valid;
          state<=WRITE;
        end
        WRITE: begin
          if(position==output_pixels-1'b1) begin
            position<=0;
            if(tap_s==S-1'b1) begin
              tap_s<=0;
              if(tap_r==R-1'b1) state<=FINISH;
              else begin tap_r<=tap_r+1'b1; state<=ISSUE; end
            end else begin tap_s<=tap_s+1'b1; state<=ISSUE; end
          end else begin position<=position+1'b1; state<=ISSUE; end
        end
        FINISH: begin done<=1; state<=IDLE; end
        default: state<=IDLE;
      endcase
    end
  end
endmodule
