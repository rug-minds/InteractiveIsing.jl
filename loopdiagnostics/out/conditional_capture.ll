; Function Signature: conditional_capture(Int64, Int64, Float64, Float64)
; Function Attrs: uwtable
define swiftcc double @julia_conditional_capture_634(ptr nonnull swiftself %pgcstack, i64 signext %"n::Int64", i64 signext %"period::Int64", double %"x::Float64", double %"bias::Float64") #0 !dbg !4 {
top:
  call void @llvm.dbg.value(metadata i64 %"n::Int64", metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i64 %"period::Int64", metadata !15, metadata !DIExpression()), !dbg !18
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
  switch i64 %"period::Int64", label %L17.preheader23 [
    i64 0, label %fail
    i64 -1, label %L17.us.preheader
  ]

L17.us.preheader:                                 ; preds = %L17.preheader
  %3 = add nsw i64 %".n::Int64", -1, !dbg !39
  %xtraiter = and i64 %".n::Int64", 7, !dbg !39
  %4 = icmp ult i64 %3, 7, !dbg !39
  br i1 %4, label %L48.loopexit24.unr-lcssa, label %L17.us.preheader.new, !dbg !39

L17.us.preheader.new:                             ; preds = %L17.us.preheader
  %unroll_iter = and i64 %".n::Int64", 9223372036854775800, !dbg !39
  br label %L17.us, !dbg !39

L17.preheader23:                                  ; preds = %L17.preheader
  %5 = add nsw i64 %".n::Int64", -1, !dbg !39
  %xtraiter27 = and i64 %".n::Int64", 7, !dbg !39
  %6 = icmp ult i64 %5, 7, !dbg !39
  br i1 %6, label %L48.loopexit.unr-lcssa, label %L17.preheader23.new, !dbg !39

L17.preheader23.new:                              ; preds = %L17.preheader23
  %unroll_iter32 = and i64 %".n::Int64", 9223372036854775800, !dbg !39
  br label %L17, !dbg !39

L17.us:                                           ; preds = %L17.us, %L17.us.preheader.new
  %value_phi6.us = phi double [ 0.000000e+00, %L17.us.preheader.new ], [ %value_phi9.us.7, %L17.us ]
  %value_phi7.us = phi double [ %"x::Float64", %L17.us.preheader.new ], [ %value_phi8.us.7, %L17.us ]
  %niter = phi i64 [ 0, %L17.us.preheader.new ], [ %niter.next.7, %L17.us ]
  %7 = fmul contract double %value_phi7.us, 0x3FF000001FF19E24, !dbg !40
  %8 = fadd contract double %7, %"bias::Float64", !dbg !40
  %9 = fcmp ule double %8, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us = select i1 %9, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us = fmul double %8, %value_phi8.p.v.us, !dbg !50
  %value_phi8.us = fadd double %value_phi7.us, %value_phi8.p.us, !dbg !50
  %10 = call double @llvm.fabs.f64(double %8), !dbg !51
  %11 = fmul contract double %10, 1.000000e-09, !dbg !51
  %value_phi9.us = fadd contract double %value_phi6.us, %11, !dbg !51
  %12 = fmul contract double %value_phi8.us, 0x3FF000001FF19E24, !dbg !40
  %13 = fadd contract double %12, %"bias::Float64", !dbg !40
  %14 = fcmp ule double %13, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.1 = select i1 %14, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.1 = fmul double %13, %value_phi8.p.v.us.1, !dbg !50
  %value_phi8.us.1 = fadd double %value_phi8.us, %value_phi8.p.us.1, !dbg !50
  %15 = call double @llvm.fabs.f64(double %13), !dbg !51
  %16 = fmul contract double %15, 1.000000e-09, !dbg !51
  %value_phi9.us.1 = fadd contract double %value_phi9.us, %16, !dbg !51
  %17 = fmul contract double %value_phi8.us.1, 0x3FF000001FF19E24, !dbg !40
  %18 = fadd contract double %17, %"bias::Float64", !dbg !40
  %19 = fcmp ule double %18, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.2 = select i1 %19, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.2 = fmul double %18, %value_phi8.p.v.us.2, !dbg !50
  %value_phi8.us.2 = fadd double %value_phi8.us.1, %value_phi8.p.us.2, !dbg !50
  %20 = call double @llvm.fabs.f64(double %18), !dbg !51
  %21 = fmul contract double %20, 1.000000e-09, !dbg !51
  %value_phi9.us.2 = fadd contract double %value_phi9.us.1, %21, !dbg !51
  %22 = fmul contract double %value_phi8.us.2, 0x3FF000001FF19E24, !dbg !40
  %23 = fadd contract double %22, %"bias::Float64", !dbg !40
  %24 = fcmp ule double %23, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.3 = select i1 %24, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.3 = fmul double %23, %value_phi8.p.v.us.3, !dbg !50
  %value_phi8.us.3 = fadd double %value_phi8.us.2, %value_phi8.p.us.3, !dbg !50
  %25 = call double @llvm.fabs.f64(double %23), !dbg !51
  %26 = fmul contract double %25, 1.000000e-09, !dbg !51
  %value_phi9.us.3 = fadd contract double %value_phi9.us.2, %26, !dbg !51
  %27 = fmul contract double %value_phi8.us.3, 0x3FF000001FF19E24, !dbg !40
  %28 = fadd contract double %27, %"bias::Float64", !dbg !40
  %29 = fcmp ule double %28, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.4 = select i1 %29, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.4 = fmul double %28, %value_phi8.p.v.us.4, !dbg !50
  %value_phi8.us.4 = fadd double %value_phi8.us.3, %value_phi8.p.us.4, !dbg !50
  %30 = call double @llvm.fabs.f64(double %28), !dbg !51
  %31 = fmul contract double %30, 1.000000e-09, !dbg !51
  %value_phi9.us.4 = fadd contract double %value_phi9.us.3, %31, !dbg !51
  %32 = fmul contract double %value_phi8.us.4, 0x3FF000001FF19E24, !dbg !40
  %33 = fadd contract double %32, %"bias::Float64", !dbg !40
  %34 = fcmp ule double %33, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.5 = select i1 %34, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.5 = fmul double %33, %value_phi8.p.v.us.5, !dbg !50
  %value_phi8.us.5 = fadd double %value_phi8.us.4, %value_phi8.p.us.5, !dbg !50
  %35 = call double @llvm.fabs.f64(double %33), !dbg !51
  %36 = fmul contract double %35, 1.000000e-09, !dbg !51
  %value_phi9.us.5 = fadd contract double %value_phi9.us.4, %36, !dbg !51
  %37 = fmul contract double %value_phi8.us.5, 0x3FF000001FF19E24, !dbg !40
  %38 = fadd contract double %37, %"bias::Float64", !dbg !40
  %39 = fcmp ule double %38, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.6 = select i1 %39, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.6 = fmul double %38, %value_phi8.p.v.us.6, !dbg !50
  %value_phi8.us.6 = fadd double %value_phi8.us.5, %value_phi8.p.us.6, !dbg !50
  %40 = call double @llvm.fabs.f64(double %38), !dbg !51
  %41 = fmul contract double %40, 1.000000e-09, !dbg !51
  %value_phi9.us.6 = fadd contract double %value_phi9.us.5, %41, !dbg !51
  %42 = fmul contract double %value_phi8.us.6, 0x3FF000001FF19E24, !dbg !40
  %43 = fadd contract double %42, %"bias::Float64", !dbg !40
  %44 = fcmp ule double %43, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.7 = select i1 %44, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.7 = fmul double %43, %value_phi8.p.v.us.7, !dbg !50
  %value_phi8.us.7 = fadd double %value_phi8.us.6, %value_phi8.p.us.7, !dbg !50
  %45 = call double @llvm.fabs.f64(double %43), !dbg !51
  %46 = fmul contract double %45, 1.000000e-09, !dbg !51
  %value_phi9.us.7 = fadd contract double %value_phi9.us.6, %46, !dbg !51
  %niter.next.7 = add i64 %niter, 8, !dbg !39
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter, !dbg !39
  br i1 %niter.ncmp.7, label %L48.loopexit24.unr-lcssa, label %L17.us, !dbg !39

L17:                                              ; preds = %L17, %L17.preheader23.new
  %value_phi4 = phi i64 [ 1, %L17.preheader23.new ], [ %110, %L17 ]
  %value_phi6 = phi double [ 0.000000e+00, %L17.preheader23.new ], [ %value_phi9.7, %L17 ]
  %value_phi7 = phi double [ %"x::Float64", %L17.preheader23.new ], [ %value_phi8.7, %L17 ]
  %niter33 = phi i64 [ 0, %L17.preheader23.new ], [ %niter33.next.7, %L17 ]
  %47 = fmul contract double %value_phi7, 0x3FF000001FF19E24, !dbg !40
  %48 = fadd contract double %47, %"bias::Float64", !dbg !40
  %49 = fcmp ule double %48, 1.250000e-01, !dbg !46
  %value_phi8.p.v = select i1 %49, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p = fmul double %48, %value_phi8.p.v, !dbg !50
  %value_phi8 = fadd double %value_phi7, %value_phi8.p, !dbg !50
  %50 = srem i64 %value_phi4, %"period::Int64", !dbg !52
  %.fr = freeze i64 %50, !dbg !54
  %.not = icmp eq i64 %.fr, 0, !dbg !54
  %51 = call double @llvm.fabs.f64(double %48), !dbg !51
  %52 = fmul contract double %51, 1.000000e-09, !dbg !51
  %53 = select i1 %.not, double %52, double -0.000000e+00, !dbg !51
  %value_phi9 = fadd contract double %value_phi6, %53, !dbg !51
  %54 = add nuw nsw i64 %value_phi4, 1, !dbg !57
  %55 = fmul contract double %value_phi8, 0x3FF000001FF19E24, !dbg !40
  %56 = fadd contract double %55, %"bias::Float64", !dbg !40
  %57 = fcmp ule double %56, 1.250000e-01, !dbg !46
  %value_phi8.p.v.1 = select i1 %57, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.1 = fmul double %56, %value_phi8.p.v.1, !dbg !50
  %value_phi8.1 = fadd double %value_phi8, %value_phi8.p.1, !dbg !50
  %58 = srem i64 %54, %"period::Int64", !dbg !52
  %.fr.1 = freeze i64 %58, !dbg !54
  %.not.1 = icmp eq i64 %.fr.1, 0, !dbg !54
  %59 = call double @llvm.fabs.f64(double %56), !dbg !51
  %60 = fmul contract double %59, 1.000000e-09, !dbg !51
  %61 = select i1 %.not.1, double %60, double -0.000000e+00, !dbg !51
  %value_phi9.1 = fadd contract double %value_phi9, %61, !dbg !51
  %62 = add nuw nsw i64 %value_phi4, 2, !dbg !57
  %63 = fmul contract double %value_phi8.1, 0x3FF000001FF19E24, !dbg !40
  %64 = fadd contract double %63, %"bias::Float64", !dbg !40
  %65 = fcmp ule double %64, 1.250000e-01, !dbg !46
  %value_phi8.p.v.2 = select i1 %65, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.2 = fmul double %64, %value_phi8.p.v.2, !dbg !50
  %value_phi8.2 = fadd double %value_phi8.1, %value_phi8.p.2, !dbg !50
  %66 = srem i64 %62, %"period::Int64", !dbg !52
  %.fr.2 = freeze i64 %66, !dbg !54
  %.not.2 = icmp eq i64 %.fr.2, 0, !dbg !54
  %67 = call double @llvm.fabs.f64(double %64), !dbg !51
  %68 = fmul contract double %67, 1.000000e-09, !dbg !51
  %69 = select i1 %.not.2, double %68, double -0.000000e+00, !dbg !51
  %value_phi9.2 = fadd contract double %value_phi9.1, %69, !dbg !51
  %70 = add nuw nsw i64 %value_phi4, 3, !dbg !57
  %71 = fmul contract double %value_phi8.2, 0x3FF000001FF19E24, !dbg !40
  %72 = fadd contract double %71, %"bias::Float64", !dbg !40
  %73 = fcmp ule double %72, 1.250000e-01, !dbg !46
  %value_phi8.p.v.3 = select i1 %73, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.3 = fmul double %72, %value_phi8.p.v.3, !dbg !50
  %value_phi8.3 = fadd double %value_phi8.2, %value_phi8.p.3, !dbg !50
  %74 = srem i64 %70, %"period::Int64", !dbg !52
  %.fr.3 = freeze i64 %74, !dbg !54
  %.not.3 = icmp eq i64 %.fr.3, 0, !dbg !54
  %75 = call double @llvm.fabs.f64(double %72), !dbg !51
  %76 = fmul contract double %75, 1.000000e-09, !dbg !51
  %77 = select i1 %.not.3, double %76, double -0.000000e+00, !dbg !51
  %value_phi9.3 = fadd contract double %value_phi9.2, %77, !dbg !51
  %78 = add nuw nsw i64 %value_phi4, 4, !dbg !57
  %79 = fmul contract double %value_phi8.3, 0x3FF000001FF19E24, !dbg !40
  %80 = fadd contract double %79, %"bias::Float64", !dbg !40
  %81 = fcmp ule double %80, 1.250000e-01, !dbg !46
  %value_phi8.p.v.4 = select i1 %81, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.4 = fmul double %80, %value_phi8.p.v.4, !dbg !50
  %value_phi8.4 = fadd double %value_phi8.3, %value_phi8.p.4, !dbg !50
  %82 = srem i64 %78, %"period::Int64", !dbg !52
  %.fr.4 = freeze i64 %82, !dbg !54
  %.not.4 = icmp eq i64 %.fr.4, 0, !dbg !54
  %83 = call double @llvm.fabs.f64(double %80), !dbg !51
  %84 = fmul contract double %83, 1.000000e-09, !dbg !51
  %85 = select i1 %.not.4, double %84, double -0.000000e+00, !dbg !51
  %value_phi9.4 = fadd contract double %value_phi9.3, %85, !dbg !51
  %86 = add nuw nsw i64 %value_phi4, 5, !dbg !57
  %87 = fmul contract double %value_phi8.4, 0x3FF000001FF19E24, !dbg !40
  %88 = fadd contract double %87, %"bias::Float64", !dbg !40
  %89 = fcmp ule double %88, 1.250000e-01, !dbg !46
  %value_phi8.p.v.5 = select i1 %89, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.5 = fmul double %88, %value_phi8.p.v.5, !dbg !50
  %value_phi8.5 = fadd double %value_phi8.4, %value_phi8.p.5, !dbg !50
  %90 = srem i64 %86, %"period::Int64", !dbg !52
  %.fr.5 = freeze i64 %90, !dbg !54
  %.not.5 = icmp eq i64 %.fr.5, 0, !dbg !54
  %91 = call double @llvm.fabs.f64(double %88), !dbg !51
  %92 = fmul contract double %91, 1.000000e-09, !dbg !51
  %93 = select i1 %.not.5, double %92, double -0.000000e+00, !dbg !51
  %value_phi9.5 = fadd contract double %value_phi9.4, %93, !dbg !51
  %94 = add nuw nsw i64 %value_phi4, 6, !dbg !57
  %95 = fmul contract double %value_phi8.5, 0x3FF000001FF19E24, !dbg !40
  %96 = fadd contract double %95, %"bias::Float64", !dbg !40
  %97 = fcmp ule double %96, 1.250000e-01, !dbg !46
  %value_phi8.p.v.6 = select i1 %97, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.6 = fmul double %96, %value_phi8.p.v.6, !dbg !50
  %value_phi8.6 = fadd double %value_phi8.5, %value_phi8.p.6, !dbg !50
  %98 = srem i64 %94, %"period::Int64", !dbg !52
  %.fr.6 = freeze i64 %98, !dbg !54
  %.not.6 = icmp eq i64 %.fr.6, 0, !dbg !54
  %99 = call double @llvm.fabs.f64(double %96), !dbg !51
  %100 = fmul contract double %99, 1.000000e-09, !dbg !51
  %101 = select i1 %.not.6, double %100, double -0.000000e+00, !dbg !51
  %value_phi9.6 = fadd contract double %value_phi9.5, %101, !dbg !51
  %102 = add nuw i64 %value_phi4, 7, !dbg !57
  %103 = fmul contract double %value_phi8.6, 0x3FF000001FF19E24, !dbg !40
  %104 = fadd contract double %103, %"bias::Float64", !dbg !40
  %105 = fcmp ule double %104, 1.250000e-01, !dbg !46
  %value_phi8.p.v.7 = select i1 %105, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.7 = fmul double %104, %value_phi8.p.v.7, !dbg !50
  %value_phi8.7 = fadd double %value_phi8.6, %value_phi8.p.7, !dbg !50
  %106 = srem i64 %102, %"period::Int64", !dbg !52
  %.fr.7 = freeze i64 %106, !dbg !54
  %.not.7 = icmp eq i64 %.fr.7, 0, !dbg !54
  %107 = call double @llvm.fabs.f64(double %104), !dbg !51
  %108 = fmul contract double %107, 1.000000e-09, !dbg !51
  %109 = select i1 %.not.7, double %108, double -0.000000e+00, !dbg !51
  %value_phi9.7 = fadd contract double %value_phi9.6, %109, !dbg !51
  %110 = add nuw i64 %value_phi4, 8, !dbg !57
  %niter33.next.7 = add i64 %niter33, 8, !dbg !39
  %niter33.ncmp.7 = icmp eq i64 %niter33.next.7, %unroll_iter32, !dbg !39
  br i1 %niter33.ncmp.7, label %L48.loopexit.unr-lcssa, label %L17, !dbg !39

L48.loopexit.unr-lcssa:                           ; preds = %L17, %L17.preheader23
  %value_phi8.lcssa.ph = phi double [ undef, %L17.preheader23 ], [ %value_phi8.7, %L17 ]
  %value_phi9.lcssa.ph = phi double [ undef, %L17.preheader23 ], [ %value_phi9.7, %L17 ]
  %value_phi4.unr = phi i64 [ 1, %L17.preheader23 ], [ %110, %L17 ]
  %value_phi6.unr = phi double [ 0.000000e+00, %L17.preheader23 ], [ %value_phi9.7, %L17 ]
  %value_phi7.unr = phi double [ %"x::Float64", %L17.preheader23 ], [ %value_phi8.7, %L17 ]
  %lcmp.mod29.not = icmp eq i64 %xtraiter27, 0, !dbg !39
  br i1 %lcmp.mod29.not, label %L48, label %L17.epil, !dbg !39

L17.epil:                                         ; preds = %L48.loopexit.unr-lcssa, %L17.epil
  %value_phi4.epil = phi i64 [ %118, %L17.epil ], [ %value_phi4.unr, %L48.loopexit.unr-lcssa ]
  %value_phi6.epil = phi double [ %value_phi9.epil, %L17.epil ], [ %value_phi6.unr, %L48.loopexit.unr-lcssa ]
  %value_phi7.epil = phi double [ %value_phi8.epil, %L17.epil ], [ %value_phi7.unr, %L48.loopexit.unr-lcssa ]
  %epil.iter28 = phi i64 [ %epil.iter28.next, %L17.epil ], [ 0, %L48.loopexit.unr-lcssa ]
  %111 = fmul contract double %value_phi7.epil, 0x3FF000001FF19E24, !dbg !40
  %112 = fadd contract double %111, %"bias::Float64", !dbg !40
  %113 = fcmp ule double %112, 1.250000e-01, !dbg !46
  %value_phi8.p.v.epil = select i1 %113, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.epil = fmul double %112, %value_phi8.p.v.epil, !dbg !50
  %value_phi8.epil = fadd double %value_phi7.epil, %value_phi8.p.epil, !dbg !50
  %114 = srem i64 %value_phi4.epil, %"period::Int64", !dbg !52
  %.fr.epil = freeze i64 %114, !dbg !54
  %.not.epil = icmp eq i64 %.fr.epil, 0, !dbg !54
  %115 = call double @llvm.fabs.f64(double %112), !dbg !51
  %116 = fmul contract double %115, 1.000000e-09, !dbg !51
  %117 = select i1 %.not.epil, double %116, double -0.000000e+00, !dbg !51
  %value_phi9.epil = fadd contract double %value_phi6.epil, %117, !dbg !51
  %118 = add nuw i64 %value_phi4.epil, 1, !dbg !57
  %epil.iter28.next = add i64 %epil.iter28, 1, !dbg !39
  %epil.iter28.cmp.not = icmp eq i64 %epil.iter28.next, %xtraiter27, !dbg !39
  br i1 %epil.iter28.cmp.not, label %L48, label %L17.epil, !dbg !39, !llvm.loop !58

L48.loopexit24.unr-lcssa:                         ; preds = %L17.us, %L17.us.preheader
  %value_phi8.us.lcssa.ph = phi double [ undef, %L17.us.preheader ], [ %value_phi8.us.7, %L17.us ]
  %value_phi9.us.lcssa.ph = phi double [ undef, %L17.us.preheader ], [ %value_phi9.us.7, %L17.us ]
  %value_phi6.us.unr = phi double [ 0.000000e+00, %L17.us.preheader ], [ %value_phi9.us.7, %L17.us ]
  %value_phi7.us.unr = phi double [ %"x::Float64", %L17.us.preheader ], [ %value_phi8.us.7, %L17.us ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !39
  br i1 %lcmp.mod.not, label %L48, label %L17.us.epil, !dbg !39

L17.us.epil:                                      ; preds = %L48.loopexit24.unr-lcssa, %L17.us.epil
  %value_phi6.us.epil = phi double [ %value_phi9.us.epil, %L17.us.epil ], [ %value_phi6.us.unr, %L48.loopexit24.unr-lcssa ]
  %value_phi7.us.epil = phi double [ %value_phi8.us.epil, %L17.us.epil ], [ %value_phi7.us.unr, %L48.loopexit24.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %L17.us.epil ], [ 0, %L48.loopexit24.unr-lcssa ]
  %119 = fmul contract double %value_phi7.us.epil, 0x3FF000001FF19E24, !dbg !40
  %120 = fadd contract double %119, %"bias::Float64", !dbg !40
  %121 = fcmp ule double %120, 1.250000e-01, !dbg !46
  %value_phi8.p.v.us.epil = select i1 %121, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !50
  %value_phi8.p.us.epil = fmul double %120, %value_phi8.p.v.us.epil, !dbg !50
  %value_phi8.us.epil = fadd double %value_phi7.us.epil, %value_phi8.p.us.epil, !dbg !50
  %122 = call double @llvm.fabs.f64(double %120), !dbg !51
  %123 = fmul contract double %122, 1.000000e-09, !dbg !51
  %value_phi9.us.epil = fadd contract double %value_phi6.us.epil, %123, !dbg !51
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !39
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !39
  br i1 %epil.iter.cmp.not, label %L48, label %L17.us.epil, !dbg !39, !llvm.loop !60

L48:                                              ; preds = %L48.loopexit24.unr-lcssa, %L17.us.epil, %L48.loopexit.unr-lcssa, %L17.epil, %top
  %value_phi13 = phi double [ 0.000000e+00, %top ], [ %value_phi9.lcssa.ph, %L48.loopexit.unr-lcssa ], [ %value_phi9.epil, %L17.epil ], [ %value_phi9.us.lcssa.ph, %L48.loopexit24.unr-lcssa ], [ %value_phi9.us.epil, %L17.us.epil ]
  %value_phi14 = phi double [ %"x::Float64", %top ], [ %value_phi8.lcssa.ph, %L48.loopexit.unr-lcssa ], [ %value_phi8.epil, %L17.epil ], [ %value_phi8.us.lcssa.ph, %L48.loopexit24.unr-lcssa ], [ %value_phi8.us.epil, %L17.us.epil ]
  %124 = fadd double %value_phi13, %value_phi14, !dbg !61
  ret double %124, !dbg !63

fail:                                             ; preds = %L17.preheader
  %jl_diverror_exception = load ptr, ptr @jl_diverror_exception, align 8, !dbg !52, !tbaa !23, !invariant.load !10, !alias.scope !64, !noalias !67, !nonnull !10
  call void @ijl_throw(ptr nonnull %jl_diverror_exception), !dbg !52
  unreachable, !dbg !52
}
