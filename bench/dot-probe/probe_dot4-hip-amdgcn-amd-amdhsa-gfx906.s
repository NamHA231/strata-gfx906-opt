	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.protected	k_dot4_builtin_chained  ; -- Begin function k_dot4_builtin_chained
	.globl	k_dot4_builtin_chained
	.p2align	8
	.type	k_dot4_builtin_chained,@function
k_dot4_builtin_chained:                 ; @k_dot4_builtin_chained
; %bb.0:
	s_load_dword s0, s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB0_5
; %bb.1:
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	s_waitcnt lgkmcnt(0)
	global_load_dword v0, v2, s[8:9]
	global_load_dword v1, v2, s[10:11]
	v_mov_b32_e32 v2, 0
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s0, s0, -1
	s_cmp_eq_u32 s0, 0
	s_waitcnt vmcnt(0)
	v_dot4_u32_u8 v2, v0, v1, v2
	s_cbranch_scc0 .LBB0_2
; %bb.3:
	s_brev_b32 s0, -2
	v_cmp_eq_u32_e32 vcc, s0, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:
	s_load_dwordx2 s[0:1], s[4:5], 0x10
	s_mov_b32 s7, 0
	s_lshl_b64 s[2:3], s[6:7], 2
	v_mov_b32_e32 v0, 0
	v_bfrev_b32_e32 v1, -2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	global_store_dword v0, v1, s[0:1]
.LBB0_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dot4_builtin_chained
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	k_dot4_builtin_chained, .Lfunc_end0-k_dot4_builtin_chained
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 144
; NumSgprs: 16
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 3
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	k_dot4_builtin_ilp4     ; -- Begin function k_dot4_builtin_ilp4
	.globl	k_dot4_builtin_ilp4
	.p2align	8
	.type	k_dot4_builtin_ilp4,@function
k_dot4_builtin_ilp4:                    ; @k_dot4_builtin_ilp4
; %bb.0:
	s_load_dword s0, s[4:5], 0x18
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB1_4
; %bb.1:
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v3, 3
	v_mov_b32_e32 v4, 2
	v_mov_b32_e32 v5, 1
	s_waitcnt lgkmcnt(0)
	global_load_dword v0, v1, s[8:9]
	global_load_dword v2, v1, s[10:11]
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v6, 1, v0
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v7, 1, v2
.LBB1_2:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s0, s0, -1
	v_dot4_u32_u8 v1, v0, v2, v1
	v_dot4_u32_u8 v5, v6, v2, v5
	v_dot4_u32_u8 v4, v0, v7, v4
	s_cmp_eq_u32 s0, 0
	v_dot4_u32_u8 v3, v6, v7, v3
	s_cbranch_scc0 .LBB1_2
; %bb.3:
	v_xor_b32_e32 v0, v5, v4
	v_xor_b32_e32 v2, v0, v3
.LBB1_4:
	v_xor_b32_e32 v0, v2, v1
	s_brev_b32 s0, -2
	v_cmp_eq_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_6
; %bb.5:
	s_load_dwordx2 s[0:1], s[4:5], 0x10
	s_mov_b32 s7, 0
	s_lshl_b64 s[2:3], s[6:7], 2
	v_mov_b32_e32 v0, 0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	global_store_dword v0, v1, s[0:1]
.LBB1_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dot4_builtin_ilp4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	k_dot4_builtin_ilp4, .Lfunc_end1-k_dot4_builtin_ilp4
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 208
; NumSgprs: 16
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	k_dot4_asm_chained      ; -- Begin function k_dot4_asm_chained
	.globl	k_dot4_asm_chained
	.p2align	8
	.type	k_dot4_asm_chained,@function
k_dot4_asm_chained:                     ; @k_dot4_asm_chained
; %bb.0:
	s_load_dword s0, s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB2_5
; %bb.1:
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	s_mov_b32 s1, 0
	s_waitcnt lgkmcnt(0)
	global_load_dword v0, v2, s[8:9]
	global_load_dword v1, v2, s[10:11]
.LBB2_2:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s0, s0, -1
	s_cmp_eq_u32 s0, 0
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot4_u32_u8 s1, v0, v1, s1
	;;#ASMEND
	s_cbranch_scc0 .LBB2_2
; %bb.3:
	s_cmp_eq_u32 s1, 0x7fffffff
	s_cbranch_scc0 .LBB2_5
; %bb.4:
	s_load_dwordx2 s[0:1], s[4:5], 0x10
	s_mov_b32 s7, 0
	s_lshl_b64 s[2:3], s[6:7], 2
	v_mov_b32_e32 v0, 0
	v_bfrev_b32_e32 v1, -2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	global_store_dword v0, v1, s[0:1]
.LBB2_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dot4_asm_chained
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end2:
	.size	k_dot4_asm_chained, .Lfunc_end2-k_dot4_asm_chained
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 140
; NumSgprs: 16
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 3
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	k_dot4_asm_ilp4         ; -- Begin function k_dot4_asm_ilp4
	.globl	k_dot4_asm_ilp4
	.p2align	8
	.type	k_dot4_asm_ilp4,@function
k_dot4_asm_ilp4:                        ; @k_dot4_asm_ilp4
; %bb.0:
	s_load_dword s1, s[4:5], 0x18
	s_mov_b32 s2, 1
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s1, 1
	s_cbranch_scc1 .LBB3_6
; %bb.1:
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	s_mov_b32 s3, 2
	s_mov_b32 s0, 0
	s_mov_b32 s7, 3
	s_waitcnt lgkmcnt(0)
	global_load_dword v0, v2, s[8:9]
	global_load_dword v1, v2, s[10:11]
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v2, 1, v0
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v3, 1, v1
.LBB3_2:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s1, s1, -1
	s_cmp_eq_u32 s1, 0
	;;#ASMSTART
	v_dot4_u32_u8 s0, v0, v1, s0
	;;#ASMEND
	;;#ASMSTART
	v_dot4_u32_u8 s2, v2, v1, s2
	;;#ASMEND
	;;#ASMSTART
	v_dot4_u32_u8 s3, v0, v3, s3
	;;#ASMEND
	;;#ASMSTART
	v_dot4_u32_u8 s7, v2, v3, s7
	;;#ASMEND
	s_cbranch_scc0 .LBB3_2
; %bb.3:
	s_xor_b32 s1, s2, s3
	s_xor_b32 s1, s1, s7
	s_xor_b32 s1, s1, s0
	s_cmp_eq_u32 s1, 0x7fffffff
	s_cbranch_scc0 .LBB3_5
.LBB3_4:
	s_load_dwordx2 s[2:3], s[4:5], 0x10
	s_mov_b32 s7, 0
	s_lshl_b64 s[4:5], s[6:7], 2
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, s0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s2, s2, s4
	s_addc_u32 s3, s3, s5
	global_store_dword v0, v1, s[2:3]
.LBB3_5:
	s_endpgm
.LBB3_6:
	s_mov_b32 s1, 0
	s_mov_b32 s0, 0
	s_xor_b32 s1, s1, s0
	s_cmp_eq_u32 s1, 0x7fffffff
	s_cbranch_scc1 .LBB3_4
	s_branch .LBB3_5
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dot4_asm_ilp4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	k_dot4_asm_ilp4, .Lfunc_end3-k_dot4_asm_ilp4
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 228
; NumSgprs: 16
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	k_fma_chained           ; -- Begin function k_fma_chained
	.globl	k_fma_chained
	.p2align	8
	.type	k_fma_chained,@function
k_fma_chained:                          ; @k_fma_chained
; %bb.0:
	s_load_dword s0, s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB4_5
; %bb.1:
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	s_waitcnt lgkmcnt(0)
	global_load_dword v0, v2, s[8:9]
	global_load_dword v1, v2, s[10:11]
	v_mov_b32_e32 v2, 0
.LBB4_2:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s0, s0, -1
	s_cmp_eq_u32 s0, 0
	s_waitcnt vmcnt(0)
	v_fma_f32 v2, v2, v0, v1
	s_cbranch_scc0 .LBB4_2
; %bb.3:
	s_mov_b32 s0, 0x4640e6b6
	v_cmp_eq_f32_e32 vcc, s0, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB4_5
; %bb.4:
	s_load_dwordx2 s[0:1], s[4:5], 0x10
	s_mov_b32 s7, 0
	s_lshl_b64 s[2:3], s[6:7], 2
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	global_store_dword v0, v1, s[0:1]
.LBB4_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_fma_chained
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end4:
	.size	k_fma_chained, .Lfunc_end4-k_fma_chained
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 148
; NumSgprs: 16
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 3
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	k_fma_ilp4              ; -- Begin function k_fma_ilp4
	.globl	k_fma_ilp4
	.p2align	8
	.type	k_fma_ilp4,@function
k_fma_ilp4:                             ; @k_fma_ilp4
; %bb.0:
	s_load_dword s0, s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB5_5
; %bb.1:
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v3, 2.0
	v_mov_b32_e32 v4, 1.0
	v_mov_b32_e32 v6, 0
	s_waitcnt lgkmcnt(0)
	global_load_dword v0, v2, s[8:9]
	global_load_dword v1, v2, s[10:11]
	v_mov_b32_e32 v2, 0x40400000
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v5, 1.0, v0
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v7, 1.0, v1
.LBB5_2:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s0, s0, -1
	v_fma_f32 v6, v6, v0, v1
	v_fma_f32 v4, v4, v5, v1
	v_fma_f32 v3, v3, v0, v7
	s_cmp_eq_u32 s0, 0
	v_fma_f32 v2, v2, v5, v7
	s_cbranch_scc0 .LBB5_2
; %bb.3:
	v_add_f32_e32 v0, v6, v4
	v_add_f32_e32 v0, v0, v3
	v_add_f32_e32 v0, v0, v2
	s_mov_b32 s0, 0x4640e6b6
	v_cmp_eq_f32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB5_5
; %bb.4:
	s_load_dwordx2 s[0:1], s[4:5], 0x10
	s_mov_b32 s7, 0
	s_lshl_b64 s[2:3], s[6:7], 2
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	global_store_dword v0, v1, s[0:1]
.LBB5_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_fma_ilp4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end5:
	.size	k_fma_ilp4, .Lfunc_end5-k_fma_ilp4
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; NumSgprs: 16
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.type	__hip_cuid_9e0acf8109acdaa7,@object ; @__hip_cuid_9e0acf8109acdaa7
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9e0acf8109acdaa7
__hip_cuid_9e0acf8109acdaa7:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9e0acf8109acdaa7, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.3 25012 e5bf7e55c91490b07c49d8960fa7983d864936c4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9e0acf8109acdaa7
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           k_dot4_builtin_chained
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         k_dot4_builtin_chained.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           k_dot4_builtin_ilp4
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         k_dot4_builtin_ilp4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           k_dot4_asm_chained
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         k_dot4_asm_chained.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           k_dot4_asm_ilp4
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         k_dot4_asm_ilp4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           k_fma_chained
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         k_fma_chained.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           k_fma_ilp4
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         k_fma_ilp4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
