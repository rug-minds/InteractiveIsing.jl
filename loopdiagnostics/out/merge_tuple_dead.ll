; Function Signature: merge_tuple_dead(Int64, Float64, Float64)
; Function Attrs: uwtable
define swiftcc double @julia_merge_tuple_dead_834(ptr nonnull swiftself %pgcstack, i64 signext %"n::Int64", double %"x::Float64", double %"bias::Float64") #0 !dbg !4 {
top:
  call void @llvm.dbg.value(metadata i64 %"n::Int64", metadata !14, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata double %"x::Float64", metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata double %"bias::Float64", metadata !16, metadata !DIExpression()), !dbg !17
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8, !tbaa !18
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8, !tbaa !22, !invariant.load !10
  fence syncscope("singlethread") seq_cst
  %1 = load volatile i64, ptr %safepoint, align 8, !dbg !17
  fence syncscope("singlethread") seq_cst
  %".n::Int64" = call i64 @llvm.smax.i64(i64 %"n::Int64", i64 0), !dbg !24
  %2 = icmp slt i64 %"n::Int64", 1, !dbg !25
  br i1 %2, label %L38, label %L16.preheader, !dbg !37

L16.preheader:                                    ; preds = %top
  %3 = add nsw i64 %".n::Int64", -1, !dbg !38
  %xtraiter = and i64 %".n::Int64", 7, !dbg !38
  %4 = icmp ult i64 %3, 7, !dbg !38
  br i1 %4, label %L38.loopexit.unr-lcssa, label %L16.preheader.new, !dbg !38

L16.preheader.new:                                ; preds = %L16.preheader
  %unroll_iter = and i64 %".n::Int64", 9223372036854775800, !dbg !38
  br label %L16, !dbg !38

L16:                                              ; preds = %L16, %L16.preheader.new
  %value_phi4 = phi double [ %"x::Float64", %L16.preheader.new ], [ %value_phi5.7, %L16 ]
  %niter = phi i64 [ 0, %L16.preheader.new ], [ %niter.next.7, %L16 ]
  %5 = fmul contract double %value_phi4, 0x3FF000001FF19E24, !dbg !39
  %6 = fadd contract double %5, %"bias::Float64", !dbg !39
  %7 = fcmp ule double %6, 1.250000e-01, !dbg !45
  %value_phi5.p.v = select i1 %7, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p = fmul double %6, %value_phi5.p.v, !dbg !49
  %value_phi5 = fadd double %value_phi4, %value_phi5.p, !dbg !49
  %8 = fmul contract double %value_phi5, 0x3FF000001FF19E24, !dbg !39
  %9 = fadd contract double %8, %"bias::Float64", !dbg !39
  %10 = fcmp ule double %9, 1.250000e-01, !dbg !45
  %value_phi5.p.v.1 = select i1 %10, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.1 = fmul double %9, %value_phi5.p.v.1, !dbg !49
  %value_phi5.1 = fadd double %value_phi5, %value_phi5.p.1, !dbg !49
  %11 = fmul contract double %value_phi5.1, 0x3FF000001FF19E24, !dbg !39
  %12 = fadd contract double %11, %"bias::Float64", !dbg !39
  %13 = fcmp ule double %12, 1.250000e-01, !dbg !45
  %value_phi5.p.v.2 = select i1 %13, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.2 = fmul double %12, %value_phi5.p.v.2, !dbg !49
  %value_phi5.2 = fadd double %value_phi5.1, %value_phi5.p.2, !dbg !49
  %14 = fmul contract double %value_phi5.2, 0x3FF000001FF19E24, !dbg !39
  %15 = fadd contract double %14, %"bias::Float64", !dbg !39
  %16 = fcmp ule double %15, 1.250000e-01, !dbg !45
  %value_phi5.p.v.3 = select i1 %16, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.3 = fmul double %15, %value_phi5.p.v.3, !dbg !49
  %value_phi5.3 = fadd double %value_phi5.2, %value_phi5.p.3, !dbg !49
  %17 = fmul contract double %value_phi5.3, 0x3FF000001FF19E24, !dbg !39
  %18 = fadd contract double %17, %"bias::Float64", !dbg !39
  %19 = fcmp ule double %18, 1.250000e-01, !dbg !45
  %value_phi5.p.v.4 = select i1 %19, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.4 = fmul double %18, %value_phi5.p.v.4, !dbg !49
  %value_phi5.4 = fadd double %value_phi5.3, %value_phi5.p.4, !dbg !49
  %20 = fmul contract double %value_phi5.4, 0x3FF000001FF19E24, !dbg !39
  %21 = fadd contract double %20, %"bias::Float64", !dbg !39
  %22 = fcmp ule double %21, 1.250000e-01, !dbg !45
  %value_phi5.p.v.5 = select i1 %22, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.5 = fmul double %21, %value_phi5.p.v.5, !dbg !49
  %value_phi5.5 = fadd double %value_phi5.4, %value_phi5.p.5, !dbg !49
  %23 = fmul contract double %value_phi5.5, 0x3FF000001FF19E24, !dbg !39
  %24 = fadd contract double %23, %"bias::Float64", !dbg !39
  %25 = fcmp ule double %24, 1.250000e-01, !dbg !45
  %value_phi5.p.v.6 = select i1 %25, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.6 = fmul double %24, %value_phi5.p.v.6, !dbg !49
  %value_phi5.6 = fadd double %value_phi5.5, %value_phi5.p.6, !dbg !49
  %26 = fmul contract double %value_phi5.6, 0x3FF000001FF19E24, !dbg !39
  %27 = fadd contract double %26, %"bias::Float64", !dbg !39
  %28 = fcmp ule double %27, 1.250000e-01, !dbg !45
  %value_phi5.p.v.7 = select i1 %28, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.7 = fmul double %27, %value_phi5.p.v.7, !dbg !49
  %value_phi5.7 = fadd double %value_phi5.6, %value_phi5.p.7, !dbg !49
  %niter.next.7 = add i64 %niter, 8, !dbg !38
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter, !dbg !38
  br i1 %niter.ncmp.7, label %L38.loopexit.unr-lcssa, label %L16, !dbg !38

L38.loopexit.unr-lcssa:                           ; preds = %L16, %L16.preheader
  %value_phi5.lcssa.ph = phi double [ undef, %L16.preheader ], [ %value_phi5.7, %L16 ]
  %value_phi4.unr = phi double [ %"x::Float64", %L16.preheader ], [ %value_phi5.7, %L16 ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !38
  br i1 %lcmp.mod.not, label %L38, label %L16.epil, !dbg !38

L16.epil:                                         ; preds = %L38.loopexit.unr-lcssa, %L16.epil
  %value_phi4.epil = phi double [ %value_phi5.epil, %L16.epil ], [ %value_phi4.unr, %L38.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %L16.epil ], [ 0, %L38.loopexit.unr-lcssa ]
  %29 = fmul contract double %value_phi4.epil, 0x3FF000001FF19E24, !dbg !39
  %30 = fadd contract double %29, %"bias::Float64", !dbg !39
  %31 = fcmp ule double %30, 1.250000e-01, !dbg !45
  %value_phi5.p.v.epil = select i1 %31, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !49
  %value_phi5.p.epil = fmul double %30, %value_phi5.p.v.epil, !dbg !49
  %value_phi5.epil = fadd double %value_phi4.epil, %value_phi5.p.epil, !dbg !49
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !38
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !38
  br i1 %epil.iter.cmp.not, label %L38, label %L16.epil, !dbg !38, !llvm.loop !50

L38:                                              ; preds = %L38.loopexit.unr-lcssa, %L16.epil, %top
  %value_phi8 = phi double [ %"x::Float64", %top ], [ %value_phi5.lcssa.ph, %L38.loopexit.unr-lcssa ], [ %value_phi5.epil, %L16.epil ]
  ret double %value_phi8, !dbg !52
}
