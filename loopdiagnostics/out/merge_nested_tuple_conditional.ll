; Function Signature: merge_nested_tuple_conditional(Int64, Base.Val{20}, Float64, Float64)
; Function Attrs: uwtable
define swiftcc double @julia_merge_nested_tuple_conditional_854(ptr nonnull swiftself %pgcstack, i64 signext %"n::Int64", double %"x::Float64", double %"bias::Float64") #0 !dbg !4 {
top:
  call void @llvm.dbg.value(metadata i64 %"n::Int64", metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata double %"x::Float64", metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata double %"bias::Float64", metadata !17, metadata !DIExpression()), !dbg !18
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8, !tbaa !19
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8, !tbaa !23, !invariant.load !10
  fence syncscope("singlethread") seq_cst
  %1 = load volatile i64, ptr %safepoint, align 8, !dbg !18
  fence syncscope("singlethread") seq_cst
  %".n::Int64" = call i64 @llvm.smax.i64(i64 %"n::Int64", i64 0), !dbg !25
  %2 = icmp slt i64 %"n::Int64", 1, !dbg !26
  br i1 %2, label %L48, label %L17.preheader, !dbg !38

L17.preheader:                                    ; preds = %top
  %3 = add nsw i64 %".n::Int64", -1, !dbg !39
  %xtraiter = and i64 %".n::Int64", 7, !dbg !39
  %4 = icmp ult i64 %3, 7, !dbg !39
  br i1 %4, label %L48.loopexit.unr-lcssa, label %L17.preheader.new, !dbg !39

L17.preheader.new:                                ; preds = %L17.preheader
  %unroll_iter = and i64 %".n::Int64", 9223372036854775800, !dbg !39
  br label %L17, !dbg !39

L17:                                              ; preds = %L17, %L17.preheader.new
  %value_phi4 = phi i64 [ 1, %L17.preheader.new ], [ %60, %L17 ]
  %value_phi6 = phi double [ 0.000000e+00, %L17.preheader.new ], [ %value_phi9.7, %L17 ]
  %value_phi7 = phi double [ %"x::Float64", %L17.preheader.new ], [ %value_phi8.7, %L17 ]
  %niter = phi i64 [ 0, %L17.preheader.new ], [ %niter.next.7, %L17 ]
  %5 = fmul contract double %value_phi7, 0x3FF000001FF19E24, !dbg !40
  %6 = fadd contract double %5, %"bias::Float64", !dbg !40
  %7 = fcmp ule double %6, 1.250000e-01, !dbg !46
  %value_phi8.p.v = select i1 %7, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p = fmul double %6, %value_phi8.p.v, !dbg !50
  %value_phi8 = fadd double %value_phi7, %value_phi8.p, !dbg !50
  %.urem = urem i64 %value_phi4, 20
  %.not = icmp eq i64 %.urem, 0, !dbg !51
  %8 = call double @llvm.fabs.f64(double %6), !dbg !54
  %9 = fmul contract double %8, 1.000000e-09, !dbg !54
  %10 = select i1 %.not, double %9, double -0.000000e+00, !dbg !54
  %value_phi9 = fadd contract double %value_phi6, %10, !dbg !54
  %11 = add nuw nsw i64 %value_phi4, 1, !dbg !55
  %12 = fmul contract double %value_phi8, 0x3FF000001FF19E24, !dbg !40
  %13 = fadd contract double %12, %"bias::Float64", !dbg !40
  %14 = fcmp ule double %13, 1.250000e-01, !dbg !46
  %value_phi8.p.v.1 = select i1 %14, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.1 = fmul double %13, %value_phi8.p.v.1, !dbg !50
  %value_phi8.1 = fadd double %value_phi8, %value_phi8.p.1, !dbg !50
  %.urem.1 = urem i64 %11, 20
  %.not.1 = icmp eq i64 %.urem.1, 0, !dbg !51
  %15 = call double @llvm.fabs.f64(double %13), !dbg !54
  %16 = fmul contract double %15, 1.000000e-09, !dbg !54
  %17 = select i1 %.not.1, double %16, double -0.000000e+00, !dbg !54
  %value_phi9.1 = fadd contract double %value_phi9, %17, !dbg !54
  %18 = add nuw nsw i64 %value_phi4, 2, !dbg !55
  %19 = fmul contract double %value_phi8.1, 0x3FF000001FF19E24, !dbg !40
  %20 = fadd contract double %19, %"bias::Float64", !dbg !40
  %21 = fcmp ule double %20, 1.250000e-01, !dbg !46
  %value_phi8.p.v.2 = select i1 %21, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.2 = fmul double %20, %value_phi8.p.v.2, !dbg !50
  %value_phi8.2 = fadd double %value_phi8.1, %value_phi8.p.2, !dbg !50
  %.urem.2 = urem i64 %18, 20
  %.not.2 = icmp eq i64 %.urem.2, 0, !dbg !51
  %22 = call double @llvm.fabs.f64(double %20), !dbg !54
  %23 = fmul contract double %22, 1.000000e-09, !dbg !54
  %24 = select i1 %.not.2, double %23, double -0.000000e+00, !dbg !54
  %value_phi9.2 = fadd contract double %value_phi9.1, %24, !dbg !54
  %25 = add nuw nsw i64 %value_phi4, 3, !dbg !55
  %26 = fmul contract double %value_phi8.2, 0x3FF000001FF19E24, !dbg !40
  %27 = fadd contract double %26, %"bias::Float64", !dbg !40
  %28 = fcmp ule double %27, 1.250000e-01, !dbg !46
  %value_phi8.p.v.3 = select i1 %28, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.3 = fmul double %27, %value_phi8.p.v.3, !dbg !50
  %value_phi8.3 = fadd double %value_phi8.2, %value_phi8.p.3, !dbg !50
  %.urem.3 = urem i64 %25, 20
  %.not.3 = icmp eq i64 %.urem.3, 0, !dbg !51
  %29 = call double @llvm.fabs.f64(double %27), !dbg !54
  %30 = fmul contract double %29, 1.000000e-09, !dbg !54
  %31 = select i1 %.not.3, double %30, double -0.000000e+00, !dbg !54
  %value_phi9.3 = fadd contract double %value_phi9.2, %31, !dbg !54
  %32 = add nuw nsw i64 %value_phi4, 4, !dbg !55
  %33 = fmul contract double %value_phi8.3, 0x3FF000001FF19E24, !dbg !40
  %34 = fadd contract double %33, %"bias::Float64", !dbg !40
  %35 = fcmp ule double %34, 1.250000e-01, !dbg !46
  %value_phi8.p.v.4 = select i1 %35, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.4 = fmul double %34, %value_phi8.p.v.4, !dbg !50
  %value_phi8.4 = fadd double %value_phi8.3, %value_phi8.p.4, !dbg !50
  %.urem.4 = urem i64 %32, 20
  %.not.4 = icmp eq i64 %.urem.4, 0, !dbg !51
  %36 = call double @llvm.fabs.f64(double %34), !dbg !54
  %37 = fmul contract double %36, 1.000000e-09, !dbg !54
  %38 = select i1 %.not.4, double %37, double -0.000000e+00, !dbg !54
  %value_phi9.4 = fadd contract double %value_phi9.3, %38, !dbg !54
  %39 = add nuw nsw i64 %value_phi4, 5, !dbg !55
  %40 = fmul contract double %value_phi8.4, 0x3FF000001FF19E24, !dbg !40
  %41 = fadd contract double %40, %"bias::Float64", !dbg !40
  %42 = fcmp ule double %41, 1.250000e-01, !dbg !46
  %value_phi8.p.v.5 = select i1 %42, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.5 = fmul double %41, %value_phi8.p.v.5, !dbg !50
  %value_phi8.5 = fadd double %value_phi8.4, %value_phi8.p.5, !dbg !50
  %.urem.5 = urem i64 %39, 20
  %.not.5 = icmp eq i64 %.urem.5, 0, !dbg !51
  %43 = call double @llvm.fabs.f64(double %41), !dbg !54
  %44 = fmul contract double %43, 1.000000e-09, !dbg !54
  %45 = select i1 %.not.5, double %44, double -0.000000e+00, !dbg !54
  %value_phi9.5 = fadd contract double %value_phi9.4, %45, !dbg !54
  %46 = add nuw nsw i64 %value_phi4, 6, !dbg !55
  %47 = fmul contract double %value_phi8.5, 0x3FF000001FF19E24, !dbg !40
  %48 = fadd contract double %47, %"bias::Float64", !dbg !40
  %49 = fcmp ule double %48, 1.250000e-01, !dbg !46
  %value_phi8.p.v.6 = select i1 %49, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.6 = fmul double %48, %value_phi8.p.v.6, !dbg !50
  %value_phi8.6 = fadd double %value_phi8.5, %value_phi8.p.6, !dbg !50
  %.urem.6 = urem i64 %46, 20
  %.not.6 = icmp eq i64 %.urem.6, 0, !dbg !51
  %50 = call double @llvm.fabs.f64(double %48), !dbg !54
  %51 = fmul contract double %50, 1.000000e-09, !dbg !54
  %52 = select i1 %.not.6, double %51, double -0.000000e+00, !dbg !54
  %value_phi9.6 = fadd contract double %value_phi9.5, %52, !dbg !54
  %53 = add nuw i64 %value_phi4, 7, !dbg !55
  %54 = fmul contract double %value_phi8.6, 0x3FF000001FF19E24, !dbg !40
  %55 = fadd contract double %54, %"bias::Float64", !dbg !40
  %56 = fcmp ule double %55, 1.250000e-01, !dbg !46
  %value_phi8.p.v.7 = select i1 %56, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.7 = fmul double %55, %value_phi8.p.v.7, !dbg !50
  %value_phi8.7 = fadd double %value_phi8.6, %value_phi8.p.7, !dbg !50
  %.urem.7 = urem i64 %53, 20
  %.not.7 = icmp eq i64 %.urem.7, 0, !dbg !51
  %57 = call double @llvm.fabs.f64(double %55), !dbg !54
  %58 = fmul contract double %57, 1.000000e-09, !dbg !54
  %59 = select i1 %.not.7, double %58, double -0.000000e+00, !dbg !54
  %value_phi9.7 = fadd contract double %value_phi9.6, %59, !dbg !54
  %60 = add nuw i64 %value_phi4, 8, !dbg !55
  %niter.next.7 = add i64 %niter, 8, !dbg !39
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter, !dbg !39
  br i1 %niter.ncmp.7, label %L48.loopexit.unr-lcssa, label %L17, !dbg !39

L48.loopexit.unr-lcssa:                           ; preds = %L17, %L17.preheader
  %value_phi8.lcssa.ph = phi double [ undef, %L17.preheader ], [ %value_phi8.7, %L17 ]
  %value_phi9.lcssa.ph = phi double [ undef, %L17.preheader ], [ %value_phi9.7, %L17 ]
  %value_phi4.unr = phi i64 [ 1, %L17.preheader ], [ %60, %L17 ]
  %value_phi6.unr = phi double [ 0.000000e+00, %L17.preheader ], [ %value_phi9.7, %L17 ]
  %value_phi7.unr = phi double [ %"x::Float64", %L17.preheader ], [ %value_phi8.7, %L17 ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !39
  br i1 %lcmp.mod.not, label %L48, label %L17.epil, !dbg !39

L17.epil:                                         ; preds = %L48.loopexit.unr-lcssa, %L17.epil
  %value_phi4.epil = phi i64 [ %67, %L17.epil ], [ %value_phi4.unr, %L48.loopexit.unr-lcssa ]
  %value_phi6.epil = phi double [ %value_phi9.epil, %L17.epil ], [ %value_phi6.unr, %L48.loopexit.unr-lcssa ]
  %value_phi7.epil = phi double [ %value_phi8.epil, %L17.epil ], [ %value_phi7.unr, %L48.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %L17.epil ], [ 0, %L48.loopexit.unr-lcssa ]
  %61 = fmul contract double %value_phi7.epil, 0x3FF000001FF19E24, !dbg !40
  %62 = fadd contract double %61, %"bias::Float64", !dbg !40
  %63 = fcmp ule double %62, 1.250000e-01, !dbg !46
  %value_phi8.p.v.epil = select i1 %63, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.epil = fmul double %62, %value_phi8.p.v.epil, !dbg !50
  %value_phi8.epil = fadd double %value_phi7.epil, %value_phi8.p.epil, !dbg !50
  %.urem.epil = urem i64 %value_phi4.epil, 20
  %.not.epil = icmp eq i64 %.urem.epil, 0, !dbg !51
  %64 = call double @llvm.fabs.f64(double %62), !dbg !54
  %65 = fmul contract double %64, 1.000000e-09, !dbg !54
  %66 = select i1 %.not.epil, double %65, double -0.000000e+00, !dbg !54
  %value_phi9.epil = fadd contract double %value_phi6.epil, %66, !dbg !54
  %67 = add nuw i64 %value_phi4.epil, 1, !dbg !55
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !39
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !39
  br i1 %epil.iter.cmp.not, label %L48, label %L17.epil, !dbg !39, !llvm.loop !56

L48:                                              ; preds = %L48.loopexit.unr-lcssa, %L17.epil, %top
  %value_phi13 = phi double [ 0.000000e+00, %top ], [ %value_phi9.lcssa.ph, %L48.loopexit.unr-lcssa ], [ %value_phi9.epil, %L17.epil ]
  %value_phi14 = phi double [ %"x::Float64", %top ], [ %value_phi8.lcssa.ph, %L48.loopexit.unr-lcssa ], [ %value_phi8.epil, %L17.epil ]
  %68 = fadd double %value_phi13, %value_phi14, !dbg !58
  ret double %68, !dbg !60
}
