module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %z_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "z_cons_prod_lock_0"}
    %z_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "z_cons_cons_lock_0"}
    %z_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "z_buff_0"} : memref<4096xbf16> 
    %z_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "z_buff_1"} : memref<4096xbf16> 
    %z_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "z_prod_lock_0"}
    %z_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "z_cons_lock_0"}
    %y_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "y_cons_buff_0"} : memref<4096xbf16> 
    %y_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "y_cons_buff_1"} : memref<4096xbf16> 
    %y_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "y_cons_prod_lock_0"}
    %y_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "y_cons_cons_lock_0"}
    %y_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "y_prod_lock_0"}
    %y_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "y_cons_lock_0"}
    %x_cons_buff_0 = aie.buffer(%tile_0_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "x_cons_buff_0"} : memref<4096xbf16> 
    %x_cons_buff_1 = aie.buffer(%tile_0_2) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "x_cons_buff_1"} : memref<4096xbf16> 
    %x_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "x_cons_prod_lock_0"}
    %x_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "x_cons_cons_lock_0"}
    %x_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "x_prod_lock_0"}
    %x_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "x_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    func.func private @saxpy_scalar(memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>) attributes {link_with = "saxpy_scalar.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%x_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i64
      cf.switch %4 : i64, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%x_cons_buff_0 : memref<4096xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%x_cons_buff_1 : memref<4096xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%x_cons_buff_0 : memref<4096xbf16>)
    ^bb6(%5: memref<4096xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%y_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i64
      cf.switch %8 : i64, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%y_cons_buff_0 : memref<4096xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%y_cons_buff_1 : memref<4096xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%y_cons_buff_0 : memref<4096xbf16>)
    ^bb10(%9: memref<4096xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%z_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i64
      cf.switch %12 : i64, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%z_buff_0 : memref<4096xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%z_buff_1 : memref<4096xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%z_buff_0 : memref<4096xbf16>)
    ^bb14(%13: memref<4096xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @saxpy_scalar(%5, %9, %13) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>) -> ()
      aie.use_lock(%x_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%y_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%z_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous0[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["saxpy_scalar.o"]}
    aie.trace.config @trace_core_1_config(%tile_0_2) packet_type = core {
      aie.trace.reg register = "Trace_Control0" value = 2038038528 mask = 2139029507 comment = "trace mode + start event + stop event"
      aie.trace.reg register = "Trace_Control1" value = 1 mask = 28703 comment = "packet ID + packet type"
      aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" value = 289 mask = 16191 comment = "port 0 ID + port 0 master/slave + port 1 ID + port 1 master/slave"
      aie.trace.reg register = "Trace_Event0" value = 388309537 mask = 2139062143 comment = "INSTR_EVENT_0 + INSTR_EVENT_1 + INSTR_VECTOR + MEMORY_STALL"
      aie.trace.reg register = "Trace_Event1" value = 1330321944 mask = 2139062143 comment = "STREAM_STALL + LOCK_STALL + PORT_RUNNING_0 + PORT_RUNNING_1"
    }
    aie.runtime_sequence(%arg0: memref<4096xbf16>, %arg1: memref<4096xbf16>, %arg2: memref<4096xbf16>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 388309537 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 1330321944 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 2048 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 65280 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      %0 = aiex.dma_configure_task_for @x_shim_alloc {
        aie.dma_bd(%arg0 : memref<4096xbf16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @y_shim_alloc {
        aie.dma_bd(%arg1 : memref<4096xbf16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @z_shim_alloc {
        aie.dma_bd(%arg2 : memref<4096xbf16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.shim_dma_allocation @x_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%x_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%x_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%x_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%x_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%x_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%x_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%y_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%y_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%y_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%y_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%y_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%y_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%z_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%z_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%z_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%z_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%z_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%z_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @y_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @z_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
