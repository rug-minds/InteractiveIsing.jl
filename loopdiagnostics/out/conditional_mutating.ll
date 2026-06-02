; Function Signature: conditional_capture_mutating(Int64, Int64, Main.Box{Float64}, Float64)
; Function Attrs: uwtable
define swiftcc double @julia_conditional_capture_mutating_714(ptr nonnull swiftself %pgcstack, i64 signext %"n::Int64", i64 signext %"period::Int64", ptr noundef nonnull align 8 dereferenceable(8) %"box::Box", double %"bias::Float64") #0 !dbg !4 {
top:
  call void @llvm.dbg.value(metadata i64 %"n::Int64", metadata !16, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %"period::Int64", metadata !17, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata ptr %"box::Box", metadata !18, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata double %"bias::Float64", metadata !19, metadata !DIExpression()), !dbg !20
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8, !tbaa !21
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8, !tbaa !25, !invariant.load !10
  fence syncscope("singlethread") seq_cst
  %1 = load volatile i64, ptr %safepoint, align 8, !dbg !20
  fence syncscope("singlethread") seq_cst
  %".n::Int64" = call i64 @llvm.smax.i64(i64 %"n::Int64", i64 0), !dbg !27
  %2 = icmp slt i64 %"n::Int64", 1, !dbg !28
  %"box::Box.x13.pre" = load double, ptr %"box::Box", align 8, !tbaa !41, !alias.scope !45, !noalias !48
  br i1 %2, label %L49, label %L17.preheader, !dbg !40

L17.preheader:                                    ; preds = %top
  switch i64 %"period::Int64", label %L17.preheader38 [
    i64 0, label %L17.preheader.split.us
    i64 -1, label %L17.us22.preheader
  ], !dbg !53

L17.us22.preheader:                               ; preds = %L17.preheader
  %3 = add nsw i64 %".n::Int64", -1, !dbg !56
  %xtraiter = and i64 %".n::Int64", 7, !dbg !56
  %4 = icmp ult i64 %3, 7, !dbg !56
  br i1 %4, label %L49.loopexit.split.loopexit39.unr-lcssa, label %L17.us22.preheader.new, !dbg !56

L17.us22.preheader.new:                           ; preds = %L17.us22.preheader
  %unroll_iter = and i64 %".n::Int64", 9223372036854775800, !dbg !56
  br label %L17.us22, !dbg !56

L17.preheader38:                                  ; preds = %L17.preheader
  %5 = add nsw i64 %".n::Int64", -1, !dbg !56
  %xtraiter42 = and i64 %".n::Int64", 7, !dbg !56
  %6 = icmp ult i64 %5, 7, !dbg !56
  br i1 %6, label %L49.loopexit.split.loopexit.unr-lcssa, label %L17.preheader38.new, !dbg !56

L17.preheader38.new:                              ; preds = %L17.preheader38
  %unroll_iter47 = and i64 %".n::Int64", 9223372036854775800, !dbg !56
  br label %L17, !dbg !56

L17.preheader.split.us:                           ; preds = %L17.preheader
  %7 = fmul contract double %"box::Box.x13.pre", 0x3FF000001FF19E24, !dbg !57
  %8 = fadd contract double %7, %"bias::Float64", !dbg !57
  %9 = fcmp ule double %8, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us = select i1 %9, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us = fmul double %8, %value_phi7.p.v.us, !dbg !67
  %value_phi7.us = fadd double %"box::Box.x13.pre", %value_phi7.p.us, !dbg !67
  store double %value_phi7.us, ptr %"box::Box", align 8, !dbg !68, !tbaa !41, !alias.scope !45, !noalias !48
  %jl_diverror_exception = load ptr, ptr @jl_diverror_exception, align 8, !dbg !53, !tbaa !25, !invariant.load !10, !alias.scope !69, !noalias !70, !nonnull !10
  call void @ijl_throw(ptr nonnull %jl_diverror_exception), !dbg !53
  unreachable, !dbg !53

L17.us22:                                         ; preds = %L17.us22, %L17.us22.preheader.new
  %value_phi717.us23 = phi double [ %"box::Box.x13.pre", %L17.us22.preheader.new ], [ %value_phi7.us28.7, %L17.us22 ], !dbg !71
  %value_phi6.us25 = phi double [ 0.000000e+00, %L17.us22.preheader.new ], [ %value_phi8.us29.7, %L17.us22 ]
  %niter = phi i64 [ 0, %L17.us22.preheader.new ], [ %niter.next.7, %L17.us22 ]
  %10 = fmul contract double %value_phi717.us23, 0x3FF000001FF19E24, !dbg !57
  %11 = fadd contract double %10, %"bias::Float64", !dbg !57
  %12 = fcmp ule double %11, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26 = select i1 %12, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27 = fmul double %11, %value_phi7.p.v.us26, !dbg !67
  %value_phi7.us28 = fadd double %value_phi717.us23, %value_phi7.p.us27, !dbg !67
  %13 = call double @llvm.fabs.f64(double %11), !dbg !55
  %14 = fmul contract double %13, 1.000000e-09, !dbg !55
  %value_phi8.us29 = fadd contract double %value_phi6.us25, %14, !dbg !55
  %15 = fmul contract double %value_phi7.us28, 0x3FF000001FF19E24, !dbg !57
  %16 = fadd contract double %15, %"bias::Float64", !dbg !57
  %17 = fcmp ule double %16, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.1 = select i1 %17, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.1 = fmul double %16, %value_phi7.p.v.us26.1, !dbg !67
  %value_phi7.us28.1 = fadd double %value_phi7.us28, %value_phi7.p.us27.1, !dbg !67
  %18 = call double @llvm.fabs.f64(double %16), !dbg !55
  %19 = fmul contract double %18, 1.000000e-09, !dbg !55
  %value_phi8.us29.1 = fadd contract double %value_phi8.us29, %19, !dbg !55
  %20 = fmul contract double %value_phi7.us28.1, 0x3FF000001FF19E24, !dbg !57
  %21 = fadd contract double %20, %"bias::Float64", !dbg !57
  %22 = fcmp ule double %21, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.2 = select i1 %22, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.2 = fmul double %21, %value_phi7.p.v.us26.2, !dbg !67
  %value_phi7.us28.2 = fadd double %value_phi7.us28.1, %value_phi7.p.us27.2, !dbg !67
  %23 = call double @llvm.fabs.f64(double %21), !dbg !55
  %24 = fmul contract double %23, 1.000000e-09, !dbg !55
  %value_phi8.us29.2 = fadd contract double %value_phi8.us29.1, %24, !dbg !55
  %25 = fmul contract double %value_phi7.us28.2, 0x3FF000001FF19E24, !dbg !57
  %26 = fadd contract double %25, %"bias::Float64", !dbg !57
  %27 = fcmp ule double %26, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.3 = select i1 %27, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.3 = fmul double %26, %value_phi7.p.v.us26.3, !dbg !67
  %value_phi7.us28.3 = fadd double %value_phi7.us28.2, %value_phi7.p.us27.3, !dbg !67
  %28 = call double @llvm.fabs.f64(double %26), !dbg !55
  %29 = fmul contract double %28, 1.000000e-09, !dbg !55
  %value_phi8.us29.3 = fadd contract double %value_phi8.us29.2, %29, !dbg !55
  %30 = fmul contract double %value_phi7.us28.3, 0x3FF000001FF19E24, !dbg !57
  %31 = fadd contract double %30, %"bias::Float64", !dbg !57
  %32 = fcmp ule double %31, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.4 = select i1 %32, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.4 = fmul double %31, %value_phi7.p.v.us26.4, !dbg !67
  %value_phi7.us28.4 = fadd double %value_phi7.us28.3, %value_phi7.p.us27.4, !dbg !67
  %33 = call double @llvm.fabs.f64(double %31), !dbg !55
  %34 = fmul contract double %33, 1.000000e-09, !dbg !55
  %value_phi8.us29.4 = fadd contract double %value_phi8.us29.3, %34, !dbg !55
  %35 = fmul contract double %value_phi7.us28.4, 0x3FF000001FF19E24, !dbg !57
  %36 = fadd contract double %35, %"bias::Float64", !dbg !57
  %37 = fcmp ule double %36, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.5 = select i1 %37, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.5 = fmul double %36, %value_phi7.p.v.us26.5, !dbg !67
  %value_phi7.us28.5 = fadd double %value_phi7.us28.4, %value_phi7.p.us27.5, !dbg !67
  %38 = call double @llvm.fabs.f64(double %36), !dbg !55
  %39 = fmul contract double %38, 1.000000e-09, !dbg !55
  %value_phi8.us29.5 = fadd contract double %value_phi8.us29.4, %39, !dbg !55
  %40 = fmul contract double %value_phi7.us28.5, 0x3FF000001FF19E24, !dbg !57
  %41 = fadd contract double %40, %"bias::Float64", !dbg !57
  %42 = fcmp ule double %41, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.6 = select i1 %42, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.6 = fmul double %41, %value_phi7.p.v.us26.6, !dbg !67
  %value_phi7.us28.6 = fadd double %value_phi7.us28.5, %value_phi7.p.us27.6, !dbg !67
  %43 = call double @llvm.fabs.f64(double %41), !dbg !55
  %44 = fmul contract double %43, 1.000000e-09, !dbg !55
  %value_phi8.us29.6 = fadd contract double %value_phi8.us29.5, %44, !dbg !55
  %45 = fmul contract double %value_phi7.us28.6, 0x3FF000001FF19E24, !dbg !57
  %46 = fadd contract double %45, %"bias::Float64", !dbg !57
  %47 = fcmp ule double %46, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.7 = select i1 %47, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.7 = fmul double %46, %value_phi7.p.v.us26.7, !dbg !67
  %value_phi7.us28.7 = fadd double %value_phi7.us28.6, %value_phi7.p.us27.7, !dbg !67
  %48 = call double @llvm.fabs.f64(double %46), !dbg !55
  %49 = fmul contract double %48, 1.000000e-09, !dbg !55
  %value_phi8.us29.7 = fadd contract double %value_phi8.us29.6, %49, !dbg !55
  %niter.next.7 = add i64 %niter, 8, !dbg !56
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter, !dbg !56
  br i1 %niter.ncmp.7, label %L49.loopexit.split.loopexit39.unr-lcssa, label %L17.us22, !dbg !56

L17:                                              ; preds = %L17, %L17.preheader38.new
  %value_phi717 = phi double [ %"box::Box.x13.pre", %L17.preheader38.new ], [ %value_phi7.7, %L17 ], !dbg !71
  %value_phi4 = phi i64 [ 1, %L17.preheader38.new ], [ %113, %L17 ]
  %value_phi6 = phi double [ 0.000000e+00, %L17.preheader38.new ], [ %value_phi8.7, %L17 ]
  %niter48 = phi i64 [ 0, %L17.preheader38.new ], [ %niter48.next.7, %L17 ]
  %50 = fmul contract double %value_phi717, 0x3FF000001FF19E24, !dbg !57
  %51 = fadd contract double %50, %"bias::Float64", !dbg !57
  %52 = fcmp ule double %51, 1.250000e-01, !dbg !63
  %value_phi7.p.v = select i1 %52, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p = fmul double %51, %value_phi7.p.v, !dbg !67
  %value_phi7 = fadd double %value_phi717, %value_phi7.p, !dbg !67
  %53 = srem i64 %value_phi4, %"period::Int64", !dbg !53
  %.fr = freeze i64 %53, !dbg !75
  %.not = icmp eq i64 %.fr, 0, !dbg !75
  %54 = call double @llvm.fabs.f64(double %51), !dbg !55
  %55 = fmul contract double %54, 1.000000e-09, !dbg !55
  %56 = select i1 %.not, double %55, double -0.000000e+00, !dbg !55
  %value_phi8 = fadd contract double %value_phi6, %56, !dbg !55
  %57 = add nuw nsw i64 %value_phi4, 1, !dbg !78
  %58 = fmul contract double %value_phi7, 0x3FF000001FF19E24, !dbg !57
  %59 = fadd contract double %58, %"bias::Float64", !dbg !57
  %60 = fcmp ule double %59, 1.250000e-01, !dbg !63
  %value_phi7.p.v.1 = select i1 %60, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.1 = fmul double %59, %value_phi7.p.v.1, !dbg !67
  %value_phi7.1 = fadd double %value_phi7, %value_phi7.p.1, !dbg !67
  %61 = srem i64 %57, %"period::Int64", !dbg !53
  %.fr.1 = freeze i64 %61, !dbg !75
  %.not.1 = icmp eq i64 %.fr.1, 0, !dbg !75
  %62 = call double @llvm.fabs.f64(double %59), !dbg !55
  %63 = fmul contract double %62, 1.000000e-09, !dbg !55
  %64 = select i1 %.not.1, double %63, double -0.000000e+00, !dbg !55
  %value_phi8.1 = fadd contract double %value_phi8, %64, !dbg !55
  %65 = add nuw nsw i64 %value_phi4, 2, !dbg !78
  %66 = fmul contract double %value_phi7.1, 0x3FF000001FF19E24, !dbg !57
  %67 = fadd contract double %66, %"bias::Float64", !dbg !57
  %68 = fcmp ule double %67, 1.250000e-01, !dbg !63
  %value_phi7.p.v.2 = select i1 %68, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.2 = fmul double %67, %value_phi7.p.v.2, !dbg !67
  %value_phi7.2 = fadd double %value_phi7.1, %value_phi7.p.2, !dbg !67
  %69 = srem i64 %65, %"period::Int64", !dbg !53
  %.fr.2 = freeze i64 %69, !dbg !75
  %.not.2 = icmp eq i64 %.fr.2, 0, !dbg !75
  %70 = call double @llvm.fabs.f64(double %67), !dbg !55
  %71 = fmul contract double %70, 1.000000e-09, !dbg !55
  %72 = select i1 %.not.2, double %71, double -0.000000e+00, !dbg !55
  %value_phi8.2 = fadd contract double %value_phi8.1, %72, !dbg !55
  %73 = add nuw nsw i64 %value_phi4, 3, !dbg !78
  %74 = fmul contract double %value_phi7.2, 0x3FF000001FF19E24, !dbg !57
  %75 = fadd contract double %74, %"bias::Float64", !dbg !57
  %76 = fcmp ule double %75, 1.250000e-01, !dbg !63
  %value_phi7.p.v.3 = select i1 %76, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.3 = fmul double %75, %value_phi7.p.v.3, !dbg !67
  %value_phi7.3 = fadd double %value_phi7.2, %value_phi7.p.3, !dbg !67
  %77 = srem i64 %73, %"period::Int64", !dbg !53
  %.fr.3 = freeze i64 %77, !dbg !75
  %.not.3 = icmp eq i64 %.fr.3, 0, !dbg !75
  %78 = call double @llvm.fabs.f64(double %75), !dbg !55
  %79 = fmul contract double %78, 1.000000e-09, !dbg !55
  %80 = select i1 %.not.3, double %79, double -0.000000e+00, !dbg !55
  %value_phi8.3 = fadd contract double %value_phi8.2, %80, !dbg !55
  %81 = add nuw nsw i64 %value_phi4, 4, !dbg !78
  %82 = fmul contract double %value_phi7.3, 0x3FF000001FF19E24, !dbg !57
  %83 = fadd contract double %82, %"bias::Float64", !dbg !57
  %84 = fcmp ule double %83, 1.250000e-01, !dbg !63
  %value_phi7.p.v.4 = select i1 %84, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.4 = fmul double %83, %value_phi7.p.v.4, !dbg !67
  %value_phi7.4 = fadd double %value_phi7.3, %value_phi7.p.4, !dbg !67
  %85 = srem i64 %81, %"period::Int64", !dbg !53
  %.fr.4 = freeze i64 %85, !dbg !75
  %.not.4 = icmp eq i64 %.fr.4, 0, !dbg !75
  %86 = call double @llvm.fabs.f64(double %83), !dbg !55
  %87 = fmul contract double %86, 1.000000e-09, !dbg !55
  %88 = select i1 %.not.4, double %87, double -0.000000e+00, !dbg !55
  %value_phi8.4 = fadd contract double %value_phi8.3, %88, !dbg !55
  %89 = add nuw nsw i64 %value_phi4, 5, !dbg !78
  %90 = fmul contract double %value_phi7.4, 0x3FF000001FF19E24, !dbg !57
  %91 = fadd contract double %90, %"bias::Float64", !dbg !57
  %92 = fcmp ule double %91, 1.250000e-01, !dbg !63
  %value_phi7.p.v.5 = select i1 %92, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.5 = fmul double %91, %value_phi7.p.v.5, !dbg !67
  %value_phi7.5 = fadd double %value_phi7.4, %value_phi7.p.5, !dbg !67
  %93 = srem i64 %89, %"period::Int64", !dbg !53
  %.fr.5 = freeze i64 %93, !dbg !75
  %.not.5 = icmp eq i64 %.fr.5, 0, !dbg !75
  %94 = call double @llvm.fabs.f64(double %91), !dbg !55
  %95 = fmul contract double %94, 1.000000e-09, !dbg !55
  %96 = select i1 %.not.5, double %95, double -0.000000e+00, !dbg !55
  %value_phi8.5 = fadd contract double %value_phi8.4, %96, !dbg !55
  %97 = add nuw nsw i64 %value_phi4, 6, !dbg !78
  %98 = fmul contract double %value_phi7.5, 0x3FF000001FF19E24, !dbg !57
  %99 = fadd contract double %98, %"bias::Float64", !dbg !57
  %100 = fcmp ule double %99, 1.250000e-01, !dbg !63
  %value_phi7.p.v.6 = select i1 %100, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.6 = fmul double %99, %value_phi7.p.v.6, !dbg !67
  %value_phi7.6 = fadd double %value_phi7.5, %value_phi7.p.6, !dbg !67
  %101 = srem i64 %97, %"period::Int64", !dbg !53
  %.fr.6 = freeze i64 %101, !dbg !75
  %.not.6 = icmp eq i64 %.fr.6, 0, !dbg !75
  %102 = call double @llvm.fabs.f64(double %99), !dbg !55
  %103 = fmul contract double %102, 1.000000e-09, !dbg !55
  %104 = select i1 %.not.6, double %103, double -0.000000e+00, !dbg !55
  %value_phi8.6 = fadd contract double %value_phi8.5, %104, !dbg !55
  %105 = add nuw i64 %value_phi4, 7, !dbg !78
  %106 = fmul contract double %value_phi7.6, 0x3FF000001FF19E24, !dbg !57
  %107 = fadd contract double %106, %"bias::Float64", !dbg !57
  %108 = fcmp ule double %107, 1.250000e-01, !dbg !63
  %value_phi7.p.v.7 = select i1 %108, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.7 = fmul double %107, %value_phi7.p.v.7, !dbg !67
  %value_phi7.7 = fadd double %value_phi7.6, %value_phi7.p.7, !dbg !67
  %109 = srem i64 %105, %"period::Int64", !dbg !53
  %.fr.7 = freeze i64 %109, !dbg !75
  %.not.7 = icmp eq i64 %.fr.7, 0, !dbg !75
  %110 = call double @llvm.fabs.f64(double %107), !dbg !55
  %111 = fmul contract double %110, 1.000000e-09, !dbg !55
  %112 = select i1 %.not.7, double %111, double -0.000000e+00, !dbg !55
  %value_phi8.7 = fadd contract double %value_phi8.6, %112, !dbg !55
  %113 = add nuw i64 %value_phi4, 8, !dbg !78
  %niter48.next.7 = add i64 %niter48, 8, !dbg !56
  %niter48.ncmp.7 = icmp eq i64 %niter48.next.7, %unroll_iter47, !dbg !56
  br i1 %niter48.ncmp.7, label %L49.loopexit.split.loopexit.unr-lcssa, label %L17, !dbg !56

L49.loopexit.split.loopexit.unr-lcssa:            ; preds = %L17, %L17.preheader38
  %value_phi7.lcssa.ph = phi double [ undef, %L17.preheader38 ], [ %value_phi7.7, %L17 ]
  %value_phi8.lcssa.ph = phi double [ undef, %L17.preheader38 ], [ %value_phi8.7, %L17 ]
  %value_phi717.unr = phi double [ %"box::Box.x13.pre", %L17.preheader38 ], [ %value_phi7.7, %L17 ]
  %value_phi4.unr = phi i64 [ 1, %L17.preheader38 ], [ %113, %L17 ]
  %value_phi6.unr = phi double [ 0.000000e+00, %L17.preheader38 ], [ %value_phi8.7, %L17 ]
  %lcmp.mod44.not = icmp eq i64 %xtraiter42, 0, !dbg !56
  br i1 %lcmp.mod44.not, label %L49.loopexit.split, label %L17.epil, !dbg !56

L17.epil:                                         ; preds = %L49.loopexit.split.loopexit.unr-lcssa, %L17.epil
  %value_phi717.epil = phi double [ %value_phi7.epil, %L17.epil ], [ %value_phi717.unr, %L49.loopexit.split.loopexit.unr-lcssa ], !dbg !71
  %value_phi4.epil = phi i64 [ %121, %L17.epil ], [ %value_phi4.unr, %L49.loopexit.split.loopexit.unr-lcssa ]
  %value_phi6.epil = phi double [ %value_phi8.epil, %L17.epil ], [ %value_phi6.unr, %L49.loopexit.split.loopexit.unr-lcssa ]
  %epil.iter43 = phi i64 [ %epil.iter43.next, %L17.epil ], [ 0, %L49.loopexit.split.loopexit.unr-lcssa ]
  %114 = fmul contract double %value_phi717.epil, 0x3FF000001FF19E24, !dbg !57
  %115 = fadd contract double %114, %"bias::Float64", !dbg !57
  %116 = fcmp ule double %115, 1.250000e-01, !dbg !63
  %value_phi7.p.v.epil = select i1 %116, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.epil = fmul double %115, %value_phi7.p.v.epil, !dbg !67
  %value_phi7.epil = fadd double %value_phi717.epil, %value_phi7.p.epil, !dbg !67
  %117 = srem i64 %value_phi4.epil, %"period::Int64", !dbg !53
  %.fr.epil = freeze i64 %117, !dbg !75
  %.not.epil = icmp eq i64 %.fr.epil, 0, !dbg !75
  %118 = call double @llvm.fabs.f64(double %115), !dbg !55
  %119 = fmul contract double %118, 1.000000e-09, !dbg !55
  %120 = select i1 %.not.epil, double %119, double -0.000000e+00, !dbg !55
  %value_phi8.epil = fadd contract double %value_phi6.epil, %120, !dbg !55
  %121 = add nuw i64 %value_phi4.epil, 1, !dbg !78
  %epil.iter43.next = add i64 %epil.iter43, 1, !dbg !56
  %epil.iter43.cmp.not = icmp eq i64 %epil.iter43.next, %xtraiter42, !dbg !56
  br i1 %epil.iter43.cmp.not, label %L49.loopexit.split, label %L17.epil, !dbg !56, !llvm.loop !79

L49.loopexit.split.loopexit39.unr-lcssa:          ; preds = %L17.us22, %L17.us22.preheader
  %value_phi7.us28.lcssa.ph = phi double [ undef, %L17.us22.preheader ], [ %value_phi7.us28.7, %L17.us22 ]
  %value_phi8.us29.lcssa.ph = phi double [ undef, %L17.us22.preheader ], [ %value_phi8.us29.7, %L17.us22 ]
  %value_phi717.us23.unr = phi double [ %"box::Box.x13.pre", %L17.us22.preheader ], [ %value_phi7.us28.7, %L17.us22 ]
  %value_phi6.us25.unr = phi double [ 0.000000e+00, %L17.us22.preheader ], [ %value_phi8.us29.7, %L17.us22 ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !56
  br i1 %lcmp.mod.not, label %L49.loopexit.split, label %L17.us22.epil, !dbg !56

L17.us22.epil:                                    ; preds = %L49.loopexit.split.loopexit39.unr-lcssa, %L17.us22.epil
  %value_phi717.us23.epil = phi double [ %value_phi7.us28.epil, %L17.us22.epil ], [ %value_phi717.us23.unr, %L49.loopexit.split.loopexit39.unr-lcssa ], !dbg !71
  %value_phi6.us25.epil = phi double [ %value_phi8.us29.epil, %L17.us22.epil ], [ %value_phi6.us25.unr, %L49.loopexit.split.loopexit39.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %L17.us22.epil ], [ 0, %L49.loopexit.split.loopexit39.unr-lcssa ]
  %122 = fmul contract double %value_phi717.us23.epil, 0x3FF000001FF19E24, !dbg !57
  %123 = fadd contract double %122, %"bias::Float64", !dbg !57
  %124 = fcmp ule double %123, 1.250000e-01, !dbg !63
  %value_phi7.p.v.us26.epil = select i1 %124, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !67
  %value_phi7.p.us27.epil = fmul double %123, %value_phi7.p.v.us26.epil, !dbg !67
  %value_phi7.us28.epil = fadd double %value_phi717.us23.epil, %value_phi7.p.us27.epil, !dbg !67
  %125 = call double @llvm.fabs.f64(double %123), !dbg !55
  %126 = fmul contract double %125, 1.000000e-09, !dbg !55
  %value_phi8.us29.epil = fadd contract double %value_phi6.us25.epil, %126, !dbg !55
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !56
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !56
  br i1 %epil.iter.cmp.not, label %L49.loopexit.split, label %L17.us22.epil, !dbg !56, !llvm.loop !81

L49.loopexit.split:                               ; preds = %L49.loopexit.split.loopexit39.unr-lcssa, %L17.us22.epil, %L49.loopexit.split.loopexit.unr-lcssa, %L17.epil
  %.us-phi33 = phi double [ %value_phi7.lcssa.ph, %L49.loopexit.split.loopexit.unr-lcssa ], [ %value_phi7.epil, %L17.epil ], [ %value_phi7.us28.lcssa.ph, %L49.loopexit.split.loopexit39.unr-lcssa ], [ %value_phi7.us28.epil, %L17.us22.epil ]
  %.us-phi34 = phi double [ %value_phi8.lcssa.ph, %L49.loopexit.split.loopexit.unr-lcssa ], [ %value_phi8.epil, %L17.epil ], [ %value_phi8.us29.lcssa.ph, %L49.loopexit.split.loopexit39.unr-lcssa ], [ %value_phi8.us29.epil, %L17.us22.epil ]
  store double %.us-phi33, ptr %"box::Box", align 8, !dbg !68, !tbaa !41, !alias.scope !45, !noalias !48
  br label %L49, !dbg !82

L49:                                              ; preds = %L49.loopexit.split, %top
  %"box::Box.x13" = phi double [ %.us-phi33, %L49.loopexit.split ], [ %"box::Box.x13.pre", %top ], !dbg !82
  %value_phi12 = phi double [ %.us-phi34, %L49.loopexit.split ], [ 0.000000e+00, %top ]
  %127 = fadd double %value_phi12, %"box::Box.x13", !dbg !84
  ret double %127, !dbg !83
}
