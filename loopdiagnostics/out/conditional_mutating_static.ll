; Function Signature: conditional_capture_mutating_static(Int64, Base.Val{20}, Main.Box{Float64}, Float64)
; Function Attrs: uwtable
define swiftcc double @julia_conditional_capture_mutating_static_754(ptr nonnull swiftself %pgcstack, i64 signext %"n::Int64", ptr noundef nonnull align 8 dereferenceable(8) %"box::Box", double %"bias::Float64") #0 !dbg !4 {
top:
  call void @llvm.dbg.value(metadata i64 %"n::Int64", metadata !17, metadata !DIExpression()), !dbg !20
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
  %3 = add nsw i64 %".n::Int64", -1, !dbg !53
  %xtraiter = and i64 %".n::Int64", 7, !dbg !53
  %4 = icmp ult i64 %3, 7, !dbg !53
  br i1 %4, label %L49.loopexit.unr-lcssa, label %L17.preheader.new, !dbg !53

L17.preheader.new:                                ; preds = %L17.preheader
  %unroll_iter = and i64 %".n::Int64", 9223372036854775800, !dbg !53
  br label %L17, !dbg !53

L17:                                              ; preds = %L17, %L17.preheader.new
  %value_phi715 = phi double [ %"box::Box.x13.pre", %L17.preheader.new ], [ %value_phi7.7, %L17 ], !dbg !54
  %value_phi4 = phi i64 [ 1, %L17.preheader.new ], [ %60, %L17 ]
  %value_phi6 = phi double [ 0.000000e+00, %L17.preheader.new ], [ %value_phi8.7, %L17 ]
  %niter = phi i64 [ 0, %L17.preheader.new ], [ %niter.next.7, %L17 ]
  %5 = fmul contract double %value_phi715, 0x3FF000001FF19E24, !dbg !60
  %6 = fadd contract double %5, %"bias::Float64", !dbg !60
  %7 = fcmp ule double %6, 1.250000e-01, !dbg !64
  %value_phi7.p.v = select i1 %7, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p = fmul double %6, %value_phi7.p.v, !dbg !68
  %value_phi7 = fadd double %value_phi715, %value_phi7.p, !dbg !68
  %.urem = urem i64 %value_phi4, 20
  %.not = icmp eq i64 %.urem, 0, !dbg !69
  %8 = call double @llvm.fabs.f64(double %6), !dbg !72
  %9 = fmul contract double %8, 1.000000e-09, !dbg !72
  %10 = select i1 %.not, double %9, double -0.000000e+00, !dbg !72
  %value_phi8 = fadd contract double %value_phi6, %10, !dbg !72
  %11 = add nuw nsw i64 %value_phi4, 1, !dbg !73
  %12 = fmul contract double %value_phi7, 0x3FF000001FF19E24, !dbg !60
  %13 = fadd contract double %12, %"bias::Float64", !dbg !60
  %14 = fcmp ule double %13, 1.250000e-01, !dbg !64
  %value_phi7.p.v.1 = select i1 %14, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.1 = fmul double %13, %value_phi7.p.v.1, !dbg !68
  %value_phi7.1 = fadd double %value_phi7, %value_phi7.p.1, !dbg !68
  %.urem.1 = urem i64 %11, 20
  %.not.1 = icmp eq i64 %.urem.1, 0, !dbg !69
  %15 = call double @llvm.fabs.f64(double %13), !dbg !72
  %16 = fmul contract double %15, 1.000000e-09, !dbg !72
  %17 = select i1 %.not.1, double %16, double -0.000000e+00, !dbg !72
  %value_phi8.1 = fadd contract double %value_phi8, %17, !dbg !72
  %18 = add nuw nsw i64 %value_phi4, 2, !dbg !73
  %19 = fmul contract double %value_phi7.1, 0x3FF000001FF19E24, !dbg !60
  %20 = fadd contract double %19, %"bias::Float64", !dbg !60
  %21 = fcmp ule double %20, 1.250000e-01, !dbg !64
  %value_phi7.p.v.2 = select i1 %21, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.2 = fmul double %20, %value_phi7.p.v.2, !dbg !68
  %value_phi7.2 = fadd double %value_phi7.1, %value_phi7.p.2, !dbg !68
  %.urem.2 = urem i64 %18, 20
  %.not.2 = icmp eq i64 %.urem.2, 0, !dbg !69
  %22 = call double @llvm.fabs.f64(double %20), !dbg !72
  %23 = fmul contract double %22, 1.000000e-09, !dbg !72
  %24 = select i1 %.not.2, double %23, double -0.000000e+00, !dbg !72
  %value_phi8.2 = fadd contract double %value_phi8.1, %24, !dbg !72
  %25 = add nuw nsw i64 %value_phi4, 3, !dbg !73
  %26 = fmul contract double %value_phi7.2, 0x3FF000001FF19E24, !dbg !60
  %27 = fadd contract double %26, %"bias::Float64", !dbg !60
  %28 = fcmp ule double %27, 1.250000e-01, !dbg !64
  %value_phi7.p.v.3 = select i1 %28, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.3 = fmul double %27, %value_phi7.p.v.3, !dbg !68
  %value_phi7.3 = fadd double %value_phi7.2, %value_phi7.p.3, !dbg !68
  %.urem.3 = urem i64 %25, 20
  %.not.3 = icmp eq i64 %.urem.3, 0, !dbg !69
  %29 = call double @llvm.fabs.f64(double %27), !dbg !72
  %30 = fmul contract double %29, 1.000000e-09, !dbg !72
  %31 = select i1 %.not.3, double %30, double -0.000000e+00, !dbg !72
  %value_phi8.3 = fadd contract double %value_phi8.2, %31, !dbg !72
  %32 = add nuw nsw i64 %value_phi4, 4, !dbg !73
  %33 = fmul contract double %value_phi7.3, 0x3FF000001FF19E24, !dbg !60
  %34 = fadd contract double %33, %"bias::Float64", !dbg !60
  %35 = fcmp ule double %34, 1.250000e-01, !dbg !64
  %value_phi7.p.v.4 = select i1 %35, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.4 = fmul double %34, %value_phi7.p.v.4, !dbg !68
  %value_phi7.4 = fadd double %value_phi7.3, %value_phi7.p.4, !dbg !68
  %.urem.4 = urem i64 %32, 20
  %.not.4 = icmp eq i64 %.urem.4, 0, !dbg !69
  %36 = call double @llvm.fabs.f64(double %34), !dbg !72
  %37 = fmul contract double %36, 1.000000e-09, !dbg !72
  %38 = select i1 %.not.4, double %37, double -0.000000e+00, !dbg !72
  %value_phi8.4 = fadd contract double %value_phi8.3, %38, !dbg !72
  %39 = add nuw nsw i64 %value_phi4, 5, !dbg !73
  %40 = fmul contract double %value_phi7.4, 0x3FF000001FF19E24, !dbg !60
  %41 = fadd contract double %40, %"bias::Float64", !dbg !60
  %42 = fcmp ule double %41, 1.250000e-01, !dbg !64
  %value_phi7.p.v.5 = select i1 %42, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.5 = fmul double %41, %value_phi7.p.v.5, !dbg !68
  %value_phi7.5 = fadd double %value_phi7.4, %value_phi7.p.5, !dbg !68
  %.urem.5 = urem i64 %39, 20
  %.not.5 = icmp eq i64 %.urem.5, 0, !dbg !69
  %43 = call double @llvm.fabs.f64(double %41), !dbg !72
  %44 = fmul contract double %43, 1.000000e-09, !dbg !72
  %45 = select i1 %.not.5, double %44, double -0.000000e+00, !dbg !72
  %value_phi8.5 = fadd contract double %value_phi8.4, %45, !dbg !72
  %46 = add nuw nsw i64 %value_phi4, 6, !dbg !73
  %47 = fmul contract double %value_phi7.5, 0x3FF000001FF19E24, !dbg !60
  %48 = fadd contract double %47, %"bias::Float64", !dbg !60
  %49 = fcmp ule double %48, 1.250000e-01, !dbg !64
  %value_phi7.p.v.6 = select i1 %49, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.6 = fmul double %48, %value_phi7.p.v.6, !dbg !68
  %value_phi7.6 = fadd double %value_phi7.5, %value_phi7.p.6, !dbg !68
  %.urem.6 = urem i64 %46, 20
  %.not.6 = icmp eq i64 %.urem.6, 0, !dbg !69
  %50 = call double @llvm.fabs.f64(double %48), !dbg !72
  %51 = fmul contract double %50, 1.000000e-09, !dbg !72
  %52 = select i1 %.not.6, double %51, double -0.000000e+00, !dbg !72
  %value_phi8.6 = fadd contract double %value_phi8.5, %52, !dbg !72
  %53 = add nuw i64 %value_phi4, 7, !dbg !73
  %54 = fmul contract double %value_phi7.6, 0x3FF000001FF19E24, !dbg !60
  %55 = fadd contract double %54, %"bias::Float64", !dbg !60
  %56 = fcmp ule double %55, 1.250000e-01, !dbg !64
  %value_phi7.p.v.7 = select i1 %56, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.7 = fmul double %55, %value_phi7.p.v.7, !dbg !68
  %value_phi7.7 = fadd double %value_phi7.6, %value_phi7.p.7, !dbg !68
  %.urem.7 = urem i64 %53, 20
  %.not.7 = icmp eq i64 %.urem.7, 0, !dbg !69
  %57 = call double @llvm.fabs.f64(double %55), !dbg !72
  %58 = fmul contract double %57, 1.000000e-09, !dbg !72
  %59 = select i1 %.not.7, double %58, double -0.000000e+00, !dbg !72
  %value_phi8.7 = fadd contract double %value_phi8.6, %59, !dbg !72
  %60 = add nuw i64 %value_phi4, 8, !dbg !73
  %niter.next.7 = add i64 %niter, 8, !dbg !53
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter, !dbg !53
  br i1 %niter.ncmp.7, label %L49.loopexit.unr-lcssa, label %L17, !dbg !53

L49.loopexit.unr-lcssa:                           ; preds = %L17, %L17.preheader
  %value_phi7.lcssa.ph = phi double [ undef, %L17.preheader ], [ %value_phi7.7, %L17 ]
  %value_phi8.lcssa.ph = phi double [ undef, %L17.preheader ], [ %value_phi8.7, %L17 ]
  %value_phi715.unr = phi double [ %"box::Box.x13.pre", %L17.preheader ], [ %value_phi7.7, %L17 ]
  %value_phi4.unr = phi i64 [ 1, %L17.preheader ], [ %60, %L17 ]
  %value_phi6.unr = phi double [ 0.000000e+00, %L17.preheader ], [ %value_phi8.7, %L17 ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !53
  br i1 %lcmp.mod.not, label %L49.loopexit, label %L17.epil, !dbg !53

L17.epil:                                         ; preds = %L49.loopexit.unr-lcssa, %L17.epil
  %value_phi715.epil = phi double [ %value_phi7.epil, %L17.epil ], [ %value_phi715.unr, %L49.loopexit.unr-lcssa ], !dbg !54
  %value_phi4.epil = phi i64 [ %67, %L17.epil ], [ %value_phi4.unr, %L49.loopexit.unr-lcssa ]
  %value_phi6.epil = phi double [ %value_phi8.epil, %L17.epil ], [ %value_phi6.unr, %L49.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %L17.epil ], [ 0, %L49.loopexit.unr-lcssa ]
  %61 = fmul contract double %value_phi715.epil, 0x3FF000001FF19E24, !dbg !60
  %62 = fadd contract double %61, %"bias::Float64", !dbg !60
  %63 = fcmp ule double %62, 1.250000e-01, !dbg !64
  %value_phi7.p.v.epil = select i1 %63, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !68
  %value_phi7.p.epil = fmul double %62, %value_phi7.p.v.epil, !dbg !68
  %value_phi7.epil = fadd double %value_phi715.epil, %value_phi7.p.epil, !dbg !68
  %.urem.epil = urem i64 %value_phi4.epil, 20
  %.not.epil = icmp eq i64 %.urem.epil, 0, !dbg !69
  %64 = call double @llvm.fabs.f64(double %62), !dbg !72
  %65 = fmul contract double %64, 1.000000e-09, !dbg !72
  %66 = select i1 %.not.epil, double %65, double -0.000000e+00, !dbg !72
  %value_phi8.epil = fadd contract double %value_phi6.epil, %66, !dbg !72
  %67 = add nuw i64 %value_phi4.epil, 1, !dbg !73
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !53
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !53
  br i1 %epil.iter.cmp.not, label %L49.loopexit, label %L17.epil, !dbg !53, !llvm.loop !74

L49.loopexit:                                     ; preds = %L17.epil, %L49.loopexit.unr-lcssa
  %value_phi7.lcssa = phi double [ %value_phi7.lcssa.ph, %L49.loopexit.unr-lcssa ], [ %value_phi7.epil, %L17.epil ], !dbg !68
  %value_phi8.lcssa = phi double [ %value_phi8.lcssa.ph, %L49.loopexit.unr-lcssa ], [ %value_phi8.epil, %L17.epil ], !dbg !72
  store double %value_phi7.lcssa, ptr %"box::Box", align 8, !dbg !76, !tbaa !41, !alias.scope !45, !noalias !48
  br label %L49, !dbg !77

L49:                                              ; preds = %L49.loopexit, %top
  %"box::Box.x13" = phi double [ %value_phi7.lcssa, %L49.loopexit ], [ %"box::Box.x13.pre", %top ], !dbg !77
  %value_phi12 = phi double [ %value_phi8.lcssa, %L49.loopexit ], [ 0.000000e+00, %top ]
  %68 = fadd double %value_phi12, %"box::Box.x13", !dbg !79
  ret double %68, !dbg !78
}
