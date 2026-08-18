module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %C_L2L3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_cons_prod_lock_0"}
    %C_L2L3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_cons_cons_lock_0"}
    %C_L1L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "C_L1L2_cons_buff_0"} : memref<4096xbf16> 
    %C_L1L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "C_L1L2_cons_buff_1"} : memref<4096xbf16> 
    %C_L1L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "C_L1L2_cons_prod_lock_0"}
    %C_L1L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_L1L2_cons_cons_lock_0"}
    %C_L1L2_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L2_buff_0"} : memref<4096xbf16> 
    %C_L1L2_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "C_L1L2_buff_1"} : memref<4096xbf16> 
    %C_L1L2_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "C_L1L2_prod_lock_0"}
    %C_L1L2_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L2_cons_lock_0"}
    %B_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "B_L3L2_cons_buff_0"} : memref<4096xbf16> 
    %B_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "B_L3L2_cons_buff_1"} : memref<4096xbf16> 
    %B_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_L3L2_cons_prod_lock_0"}
    %B_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_L3L2_cons_cons_lock_0"}
    %B_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_prod_lock_0"}
    %B_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_cons_lock_0"}
    %B_L2L1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "B_L2L1_cons_buff_0"} : memref<4096xbf16> 
    %B_L2L1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_cons_buff_1"} : memref<4096xbf16> 
    %B_L2L1_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_L2L1_cons_prod_lock_0"}
    %B_L2L1_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L2L1_cons_cons_lock_0"}
    %A_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "A_L3L2_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "A_L3L2_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_cons_prod_lock_0"}
    %A_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_cons_lock_0"}
    %A_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_prod_lock_0"}
    %A_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_lock_0"}
    %A_L2L1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "A_L2L1_cons_buff_0"} : memref<4096xbf16> 
    %A_L2L1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_cons_buff_1"} : memref<4096xbf16> 
    %A_L2L1_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_cons_prod_lock_0"}
    %A_L2L1_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_cons_cons_lock_0"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    func.func private @d8991b41_matmul_bf16_bf16(memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>) attributes {link_with = "matmul_bf16_bf16_d8991b41.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c16 = arith.constant 16 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb23
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb24
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb22
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb23
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_L1L2_prod_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i64
      cf.switch %6 : i64, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_buff_0 : memref<4096xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_buff_1 : memref<4096xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%C_L1L2_buff_0 : memref<4096xbf16>)
    ^bb8(%7: memref<4096xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c4096 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      memref.store %cst, %7[%8] : memref<4096xbf16>
      %10 = arith.addi %8, %c1 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb21
      %12 = arith.cmpi slt, %11, %c4 : index
      cf.cond_br %12, ^bb13, ^bb22
    ^bb13:  // pred: ^bb12
      aie.use_lock(%A_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %13 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.index_cast %14 : index to i64
      cf.switch %15 : i64, [
        default: ^bb16,
        0: ^bb14,
        1: ^bb15
      ]
    ^bb14:  // pred: ^bb13
      cf.br ^bb17(%A_L2L1_cons_buff_0 : memref<4096xbf16>)
    ^bb15:  // pred: ^bb13
      cf.br ^bb17(%A_L2L1_cons_buff_1 : memref<4096xbf16>)
    ^bb16:  // pred: ^bb13
      cf.br ^bb17(%A_L2L1_cons_buff_0 : memref<4096xbf16>)
    ^bb17(%16: memref<4096xbf16>):  // 3 preds: ^bb14, ^bb15, ^bb16
      aie.use_lock(%B_L2L1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i64
      cf.switch %19 : i64, [
        default: ^bb20,
        0: ^bb18,
        1: ^bb19
      ]
    ^bb18:  // pred: ^bb17
      cf.br ^bb21(%B_L2L1_cons_buff_0 : memref<4096xbf16>)
    ^bb19:  // pred: ^bb17
      cf.br ^bb21(%B_L2L1_cons_buff_1 : memref<4096xbf16>)
    ^bb20:  // pred: ^bb17
      cf.br ^bb21(%B_L2L1_cons_buff_0 : memref<4096xbf16>)
    ^bb21(%20: memref<4096xbf16>):  // 3 preds: ^bb18, ^bb19, ^bb20
      func.call @d8991b41_matmul_bf16_bf16(%16, %20, %7) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>) -> ()
      aie.use_lock(%A_L2L1_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%B_L2L1_cons_prod_lock_0, Release, 1)
      %26 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c2_i32 : i32
      %29 = arith.subi %27, %c2_i32 : i32
      %30 = arith.select %28, %29, %27 : i32
      memref.store %30, %_anonymous0[%c2] : memref<3xi32>
      %31 = arith.addi %11, %c1 : index
      cf.br ^bb12(%31 : index)
    ^bb22:  // pred: ^bb12
      aie.use_lock(%C_L1L2_cons_lock_0, Release, 1)
      %32 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %33 = arith.addi %32, %c1_i32 : i32
      %34 = arith.cmpi sge, %33, %c2_i32 : i32
      %35 = arith.subi %33, %c2_i32 : i32
      %36 = arith.select %34, %35, %33 : i32
      memref.store %36, %_anonymous0[%c0] : memref<3xi32>
      %37 = arith.addi %2, %c1 : index
      cf.br ^bb3(%37 : index)
    ^bb23:  // pred: ^bb3
      %38 = arith.addi %0, %c1 : index
      cf.br ^bb1(%38 : index)
    ^bb24:  // pred: ^bb1
      aie.end
    } {link_files = ["matmul_bf16_bf16_d8991b41.o"]}
    aie.trace.config @trace_core_1_config(%tile_0_2) packet_type = core {
      aie.trace.reg register = "Trace_Control0" value = 2038038528 mask = 2139029507 comment = "trace mode + start event + stop event"
      aie.trace.reg register = "Trace_Control1" value = 1 mask = 28703 comment = "packet ID + packet type"
      aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" value = 289 mask = 16191 comment = "port 0 ID + port 0 master/slave + port 1 ID + port 1 master/slave"
      aie.trace.reg register = "Trace_Event0" value = 388309537 mask = 2139062143 comment = "INSTR_EVENT_0 + INSTR_EVENT_1 + INSTR_VECTOR + MEMORY_STALL"
      aie.trace.reg register = "Trace_Event1" value = 1330321944 mask = 2139062143 comment = "STREAM_STALL + LOCK_STALL + PORT_RUNNING_0 + PORT_RUNNING_1"
    }
    aie.runtime_sequence(%arg0: memref<256x256xbf16>, %arg1: memref<256x256xbf16>, %arg2: memref<256x256xbf16>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 388309537 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 1330321944 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 65280 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      %0 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<256x256xbf16>, 0, 16384, [<size = 4, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<256x256xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 256, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
        aie.dma_bd(%arg2 : memref<256x256xbf16>, 0, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      %3 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<256x256xbf16>, 16384, 16384, [<size = 4, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<256x256xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 256, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
        aie.dma_bd(%arg2 : memref<256x256xbf16>, 16384, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      aiex.dma_await_task(%5)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      %6 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<256x256xbf16>, 32768, 16384, [<size = 4, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<256x256xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 256, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
        aie.dma_bd(%arg2 : memref<256x256xbf16>, 32768, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      aiex.dma_await_task(%8)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      %9 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<256x256xbf16>, 49152, 16384, [<size = 4, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2_shim_alloc {
        aie.dma_bd(%arg1 : memref<256x256xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 256, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_shim_alloc {
        aie.dma_bd(%arg2 : memref<256x256xbf16>, 49152, 16384, [<size = 1, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      aiex.dma_await_task(%11)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L1L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L1L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L1L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L1L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L1L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L1L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L1L2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L1L2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @C_L2L3_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
