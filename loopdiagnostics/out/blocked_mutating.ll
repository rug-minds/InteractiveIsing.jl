; Function Signature: blocked_capture_mutating(Int64, Int64, Main.Box{Float64}, Float64)
; Function Attrs: uwtable
define swiftcc double @julia_blocked_capture_mutating_734(ptr nonnull swiftself %pgcstack, i64 signext %"nblocks::Int64", i64 signext %"period::Int64", ptr noundef nonnull align 8 dereferenceable(8) %"box::Box", double %"bias::Float64") #0 !dbg !4 {
top:
  call void @llvm.dbg.value(metadata i64 %"nblocks::Int64", metadata !16, metadata !DIExpression()), !dbg !20
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
  %".nblocks::Int64" = call i64 @llvm.smax.i64(i64 %"nblocks::Int64", i64 0), !dbg !27
  %2 = icmp slt i64 %"nblocks::Int64", 1, !dbg !28
  br i1 %2, label %top.L81_crit_edge, label %L16.preheader, !dbg !40

top.L81_crit_edge:                                ; preds = %top
  %"box::Box.x17.pre" = load double, ptr %"box::Box", align 8, !dbg !41, !tbaa !45, !alias.scope !49, !noalias !52
  br label %L81, !dbg !40

L16.preheader:                                    ; preds = %top
  %3 = add i64 %"period::Int64", -1
  %. = call i64 @llvm.smax.i64(i64 %3, i64 0)
  %4 = icmp slt i64 %3, 1
  %.promoted20 = load double, ptr %"box::Box", align 8, !tbaa !45, !alias.scope !49, !noalias !52
  br i1 %4, label %L16.us.preheader, label %L16.preheader29, !dbg !57

L16.preheader29:                                  ; preds = %L16.preheader
  %5 = add nsw i64 %., -1, !dbg !58
  %6 = add nsw i64 %".nblocks::Int64", -1, !dbg !58
  %xtraiter33 = and i64 %".nblocks::Int64", 3, !dbg !58
  %7 = icmp ult i64 %6, 3, !dbg !58
  br i1 %7, label %L81.loopexit.loopexit30.unr-lcssa, label %L16.preheader29.new, !dbg !58

L16.preheader29.new:                              ; preds = %L16.preheader29
  %unroll_iter44 = and i64 %".nblocks::Int64", 9223372036854775804, !dbg !58
  br label %L16, !dbg !58

L16.us.preheader:                                 ; preds = %L16.preheader
  %8 = add nsw i64 %".nblocks::Int64", -1, !dbg !59
  %xtraiter55 = and i64 %".nblocks::Int64", 7, !dbg !59
  %9 = icmp ult i64 %8, 7, !dbg !59
  br i1 %9, label %L81.loopexit.loopexit.unr-lcssa, label %L16.us.preheader.new, !dbg !59

L16.us.preheader.new:                             ; preds = %L16.us.preheader
  %unroll_iter60 = and i64 %".nblocks::Int64", 9223372036854775800, !dbg !59
  br label %L16.us, !dbg !59

L16.us:                                           ; preds = %L16.us, %L16.us.preheader.new
  %value_phi1321.us = phi double [ %.promoted20, %L16.us.preheader.new ], [ %value_phi13.us.7, %L16.us ]
  %value_phi4.us = phi double [ 0.000000e+00, %L16.us.preheader.new ], [ %57, %L16.us ]
  %niter61 = phi i64 [ 0, %L16.us.preheader.new ], [ %niter61.next.7, %L16.us ]
  %10 = fmul contract double %value_phi1321.us, 0x3FF000001FF19E24, !dbg !60
  %11 = fadd contract double %10, %"bias::Float64", !dbg !60
  %12 = fcmp ule double %11, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us = select i1 %12, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us = fmul double %11, %value_phi13.p.v.us, !dbg !70
  %value_phi13.us = fadd double %value_phi1321.us, %value_phi13.p.us, !dbg !70
  %13 = call double @llvm.fabs.f64(double %11), !dbg !71
  %14 = fmul contract double %13, 1.000000e-09, !dbg !76
  %15 = fadd contract double %value_phi4.us, %14, !dbg !76
  %16 = fmul contract double %value_phi13.us, 0x3FF000001FF19E24, !dbg !60
  %17 = fadd contract double %16, %"bias::Float64", !dbg !60
  %18 = fcmp ule double %17, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.1 = select i1 %18, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.1 = fmul double %17, %value_phi13.p.v.us.1, !dbg !70
  %value_phi13.us.1 = fadd double %value_phi13.us, %value_phi13.p.us.1, !dbg !70
  %19 = call double @llvm.fabs.f64(double %17), !dbg !71
  %20 = fmul contract double %19, 1.000000e-09, !dbg !76
  %21 = fadd contract double %15, %20, !dbg !76
  %22 = fmul contract double %value_phi13.us.1, 0x3FF000001FF19E24, !dbg !60
  %23 = fadd contract double %22, %"bias::Float64", !dbg !60
  %24 = fcmp ule double %23, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.2 = select i1 %24, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.2 = fmul double %23, %value_phi13.p.v.us.2, !dbg !70
  %value_phi13.us.2 = fadd double %value_phi13.us.1, %value_phi13.p.us.2, !dbg !70
  %25 = call double @llvm.fabs.f64(double %23), !dbg !71
  %26 = fmul contract double %25, 1.000000e-09, !dbg !76
  %27 = fadd contract double %21, %26, !dbg !76
  %28 = fmul contract double %value_phi13.us.2, 0x3FF000001FF19E24, !dbg !60
  %29 = fadd contract double %28, %"bias::Float64", !dbg !60
  %30 = fcmp ule double %29, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.3 = select i1 %30, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.3 = fmul double %29, %value_phi13.p.v.us.3, !dbg !70
  %value_phi13.us.3 = fadd double %value_phi13.us.2, %value_phi13.p.us.3, !dbg !70
  %31 = call double @llvm.fabs.f64(double %29), !dbg !71
  %32 = fmul contract double %31, 1.000000e-09, !dbg !76
  %33 = fadd contract double %27, %32, !dbg !76
  %34 = fmul contract double %value_phi13.us.3, 0x3FF000001FF19E24, !dbg !60
  %35 = fadd contract double %34, %"bias::Float64", !dbg !60
  %36 = fcmp ule double %35, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.4 = select i1 %36, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.4 = fmul double %35, %value_phi13.p.v.us.4, !dbg !70
  %value_phi13.us.4 = fadd double %value_phi13.us.3, %value_phi13.p.us.4, !dbg !70
  %37 = call double @llvm.fabs.f64(double %35), !dbg !71
  %38 = fmul contract double %37, 1.000000e-09, !dbg !76
  %39 = fadd contract double %33, %38, !dbg !76
  %40 = fmul contract double %value_phi13.us.4, 0x3FF000001FF19E24, !dbg !60
  %41 = fadd contract double %40, %"bias::Float64", !dbg !60
  %42 = fcmp ule double %41, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.5 = select i1 %42, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.5 = fmul double %41, %value_phi13.p.v.us.5, !dbg !70
  %value_phi13.us.5 = fadd double %value_phi13.us.4, %value_phi13.p.us.5, !dbg !70
  %43 = call double @llvm.fabs.f64(double %41), !dbg !71
  %44 = fmul contract double %43, 1.000000e-09, !dbg !76
  %45 = fadd contract double %39, %44, !dbg !76
  %46 = fmul contract double %value_phi13.us.5, 0x3FF000001FF19E24, !dbg !60
  %47 = fadd contract double %46, %"bias::Float64", !dbg !60
  %48 = fcmp ule double %47, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.6 = select i1 %48, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.6 = fmul double %47, %value_phi13.p.v.us.6, !dbg !70
  %value_phi13.us.6 = fadd double %value_phi13.us.5, %value_phi13.p.us.6, !dbg !70
  %49 = call double @llvm.fabs.f64(double %47), !dbg !71
  %50 = fmul contract double %49, 1.000000e-09, !dbg !76
  %51 = fadd contract double %45, %50, !dbg !76
  %52 = fmul contract double %value_phi13.us.6, 0x3FF000001FF19E24, !dbg !60
  %53 = fadd contract double %52, %"bias::Float64", !dbg !60
  %54 = fcmp ule double %53, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.7 = select i1 %54, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.7 = fmul double %53, %value_phi13.p.v.us.7, !dbg !70
  %value_phi13.us.7 = fadd double %value_phi13.us.6, %value_phi13.p.us.7, !dbg !70
  %55 = call double @llvm.fabs.f64(double %53), !dbg !71
  %56 = fmul contract double %55, 1.000000e-09, !dbg !76
  %57 = fadd contract double %51, %56, !dbg !76
  %niter61.next.7 = add i64 %niter61, 8, !dbg !59
  %niter61.ncmp.7 = icmp eq i64 %niter61.next.7, %unroll_iter60, !dbg !59
  br i1 %niter61.ncmp.7, label %L81.loopexit.loopexit.unr-lcssa, label %L16.us, !dbg !59

L16:                                              ; preds = %L57.loopexit.3, %L16.preheader29.new
  %value_phi1321 = phi double [ %.promoted20, %L16.preheader29.new ], [ %value_phi13.3, %L57.loopexit.3 ]
  %value_phi4 = phi double [ 0.000000e+00, %L16.preheader29.new ], [ %190, %L57.loopexit.3 ]
  %niter45 = phi i64 [ 0, %L16.preheader29.new ], [ %niter45.next.3, %L57.loopexit.3 ]
  %xtraiter = and i64 %., 7, !dbg !58
  %58 = icmp ult i64 %5, 7, !dbg !58
  br i1 %58, label %L57.loopexit.unr-lcssa, label %L16.new, !dbg !58

L16.new:                                          ; preds = %L16
  %unroll_iter = and i64 %., 9223372036854775800, !dbg !58
  br label %L34, !dbg !58

L34:                                              ; preds = %L34, %L16.new
  %value_phi919 = phi double [ %value_phi1321, %L16.new ], [ %value_phi9.7, %L34 ], !dbg !77
  %niter = phi i64 [ 0, %L16.new ], [ %niter.next.7, %L34 ]
  %59 = fmul contract double %value_phi919, 0x3FF000001FF19E24, !dbg !80
  %60 = fadd contract double %59, %"bias::Float64", !dbg !80
  %61 = fcmp ule double %60, 1.250000e-01, !dbg !82
  %value_phi9.p.v = select i1 %61, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p = fmul double %60, %value_phi9.p.v, !dbg !85
  %value_phi9 = fadd double %value_phi919, %value_phi9.p, !dbg !85
  %62 = fmul contract double %value_phi9, 0x3FF000001FF19E24, !dbg !80
  %63 = fadd contract double %62, %"bias::Float64", !dbg !80
  %64 = fcmp ule double %63, 1.250000e-01, !dbg !82
  %value_phi9.p.v.1 = select i1 %64, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.1 = fmul double %63, %value_phi9.p.v.1, !dbg !85
  %value_phi9.1 = fadd double %value_phi9, %value_phi9.p.1, !dbg !85
  %65 = fmul contract double %value_phi9.1, 0x3FF000001FF19E24, !dbg !80
  %66 = fadd contract double %65, %"bias::Float64", !dbg !80
  %67 = fcmp ule double %66, 1.250000e-01, !dbg !82
  %value_phi9.p.v.2 = select i1 %67, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.2 = fmul double %66, %value_phi9.p.v.2, !dbg !85
  %value_phi9.2 = fadd double %value_phi9.1, %value_phi9.p.2, !dbg !85
  %68 = fmul contract double %value_phi9.2, 0x3FF000001FF19E24, !dbg !80
  %69 = fadd contract double %68, %"bias::Float64", !dbg !80
  %70 = fcmp ule double %69, 1.250000e-01, !dbg !82
  %value_phi9.p.v.3 = select i1 %70, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.3 = fmul double %69, %value_phi9.p.v.3, !dbg !85
  %value_phi9.3 = fadd double %value_phi9.2, %value_phi9.p.3, !dbg !85
  %71 = fmul contract double %value_phi9.3, 0x3FF000001FF19E24, !dbg !80
  %72 = fadd contract double %71, %"bias::Float64", !dbg !80
  %73 = fcmp ule double %72, 1.250000e-01, !dbg !82
  %value_phi9.p.v.4 = select i1 %73, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.4 = fmul double %72, %value_phi9.p.v.4, !dbg !85
  %value_phi9.4 = fadd double %value_phi9.3, %value_phi9.p.4, !dbg !85
  %74 = fmul contract double %value_phi9.4, 0x3FF000001FF19E24, !dbg !80
  %75 = fadd contract double %74, %"bias::Float64", !dbg !80
  %76 = fcmp ule double %75, 1.250000e-01, !dbg !82
  %value_phi9.p.v.5 = select i1 %76, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.5 = fmul double %75, %value_phi9.p.v.5, !dbg !85
  %value_phi9.5 = fadd double %value_phi9.4, %value_phi9.p.5, !dbg !85
  %77 = fmul contract double %value_phi9.5, 0x3FF000001FF19E24, !dbg !80
  %78 = fadd contract double %77, %"bias::Float64", !dbg !80
  %79 = fcmp ule double %78, 1.250000e-01, !dbg !82
  %value_phi9.p.v.6 = select i1 %79, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.6 = fmul double %78, %value_phi9.p.v.6, !dbg !85
  %value_phi9.6 = fadd double %value_phi9.5, %value_phi9.p.6, !dbg !85
  %80 = fmul contract double %value_phi9.6, 0x3FF000001FF19E24, !dbg !80
  %81 = fadd contract double %80, %"bias::Float64", !dbg !80
  %82 = fcmp ule double %81, 1.250000e-01, !dbg !82
  %value_phi9.p.v.7 = select i1 %82, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.7 = fmul double %81, %value_phi9.p.v.7, !dbg !85
  %value_phi9.7 = fadd double %value_phi9.6, %value_phi9.p.7, !dbg !85
  %niter.next.7 = add i64 %niter, 8, !dbg !58
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter, !dbg !58
  br i1 %niter.ncmp.7, label %L57.loopexit.unr-lcssa, label %L34, !dbg !58

L57.loopexit.unr-lcssa:                           ; preds = %L34, %L16
  %value_phi9.lcssa.ph = phi double [ undef, %L16 ], [ %value_phi9.7, %L34 ]
  %value_phi919.unr = phi double [ %value_phi1321, %L16 ], [ %value_phi9.7, %L34 ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !58
  br i1 %lcmp.mod.not, label %L57.loopexit, label %L34.epil, !dbg !58

L34.epil:                                         ; preds = %L57.loopexit.unr-lcssa, %L34.epil
  %value_phi919.epil = phi double [ %value_phi9.epil, %L34.epil ], [ %value_phi919.unr, %L57.loopexit.unr-lcssa ], !dbg !77
  %epil.iter = phi i64 [ %epil.iter.next, %L34.epil ], [ 0, %L57.loopexit.unr-lcssa ]
  %83 = fmul contract double %value_phi919.epil, 0x3FF000001FF19E24, !dbg !80
  %84 = fadd contract double %83, %"bias::Float64", !dbg !80
  %85 = fcmp ule double %84, 1.250000e-01, !dbg !82
  %value_phi9.p.v.epil = select i1 %85, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.epil = fmul double %84, %value_phi9.p.v.epil, !dbg !85
  %value_phi9.epil = fadd double %value_phi919.epil, %value_phi9.p.epil, !dbg !85
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !58
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !58
  br i1 %epil.iter.cmp.not, label %L57.loopexit, label %L34.epil, !dbg !58, !llvm.loop !86

L57.loopexit:                                     ; preds = %L34.epil, %L57.loopexit.unr-lcssa
  %value_phi9.lcssa = phi double [ %value_phi9.lcssa.ph, %L57.loopexit.unr-lcssa ], [ %value_phi9.epil, %L34.epil ], !dbg !85
  %86 = fmul contract double %value_phi9.lcssa, 0x3FF000001FF19E24, !dbg !60
  %87 = fadd contract double %86, %"bias::Float64", !dbg !60
  %88 = fcmp ule double %87, 1.250000e-01, !dbg !66
  %value_phi13.p.v = select i1 %88, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p = fmul double %87, %value_phi13.p.v, !dbg !70
  %value_phi13 = fadd double %value_phi9.lcssa, %value_phi13.p, !dbg !70
  %89 = call double @llvm.fabs.f64(double %87), !dbg !71
  %90 = fmul contract double %89, 1.000000e-09, !dbg !76
  %91 = fadd contract double %value_phi4, %90, !dbg !76
  br i1 %58, label %L57.loopexit.unr-lcssa.1, label %L16.new.1, !dbg !58

L16.new.1:                                        ; preds = %L57.loopexit
  %unroll_iter.1 = and i64 %., 9223372036854775800, !dbg !58
  br label %L34.1, !dbg !58

L34.1:                                            ; preds = %L34.1, %L16.new.1
  %value_phi919.1 = phi double [ %value_phi13, %L16.new.1 ], [ %value_phi9.7.1, %L34.1 ], !dbg !77
  %niter.1 = phi i64 [ 0, %L16.new.1 ], [ %niter.next.7.1, %L34.1 ]
  %92 = fmul contract double %value_phi919.1, 0x3FF000001FF19E24, !dbg !80
  %93 = fadd contract double %92, %"bias::Float64", !dbg !80
  %94 = fcmp ule double %93, 1.250000e-01, !dbg !82
  %value_phi9.p.v.146 = select i1 %94, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.147 = fmul double %93, %value_phi9.p.v.146, !dbg !85
  %value_phi9.148 = fadd double %value_phi919.1, %value_phi9.p.147, !dbg !85
  %95 = fmul contract double %value_phi9.148, 0x3FF000001FF19E24, !dbg !80
  %96 = fadd contract double %95, %"bias::Float64", !dbg !80
  %97 = fcmp ule double %96, 1.250000e-01, !dbg !82
  %value_phi9.p.v.1.1 = select i1 %97, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.1.1 = fmul double %96, %value_phi9.p.v.1.1, !dbg !85
  %value_phi9.1.1 = fadd double %value_phi9.148, %value_phi9.p.1.1, !dbg !85
  %98 = fmul contract double %value_phi9.1.1, 0x3FF000001FF19E24, !dbg !80
  %99 = fadd contract double %98, %"bias::Float64", !dbg !80
  %100 = fcmp ule double %99, 1.250000e-01, !dbg !82
  %value_phi9.p.v.2.1 = select i1 %100, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.2.1 = fmul double %99, %value_phi9.p.v.2.1, !dbg !85
  %value_phi9.2.1 = fadd double %value_phi9.1.1, %value_phi9.p.2.1, !dbg !85
  %101 = fmul contract double %value_phi9.2.1, 0x3FF000001FF19E24, !dbg !80
  %102 = fadd contract double %101, %"bias::Float64", !dbg !80
  %103 = fcmp ule double %102, 1.250000e-01, !dbg !82
  %value_phi9.p.v.3.1 = select i1 %103, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.3.1 = fmul double %102, %value_phi9.p.v.3.1, !dbg !85
  %value_phi9.3.1 = fadd double %value_phi9.2.1, %value_phi9.p.3.1, !dbg !85
  %104 = fmul contract double %value_phi9.3.1, 0x3FF000001FF19E24, !dbg !80
  %105 = fadd contract double %104, %"bias::Float64", !dbg !80
  %106 = fcmp ule double %105, 1.250000e-01, !dbg !82
  %value_phi9.p.v.4.1 = select i1 %106, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.4.1 = fmul double %105, %value_phi9.p.v.4.1, !dbg !85
  %value_phi9.4.1 = fadd double %value_phi9.3.1, %value_phi9.p.4.1, !dbg !85
  %107 = fmul contract double %value_phi9.4.1, 0x3FF000001FF19E24, !dbg !80
  %108 = fadd contract double %107, %"bias::Float64", !dbg !80
  %109 = fcmp ule double %108, 1.250000e-01, !dbg !82
  %value_phi9.p.v.5.1 = select i1 %109, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.5.1 = fmul double %108, %value_phi9.p.v.5.1, !dbg !85
  %value_phi9.5.1 = fadd double %value_phi9.4.1, %value_phi9.p.5.1, !dbg !85
  %110 = fmul contract double %value_phi9.5.1, 0x3FF000001FF19E24, !dbg !80
  %111 = fadd contract double %110, %"bias::Float64", !dbg !80
  %112 = fcmp ule double %111, 1.250000e-01, !dbg !82
  %value_phi9.p.v.6.1 = select i1 %112, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.6.1 = fmul double %111, %value_phi9.p.v.6.1, !dbg !85
  %value_phi9.6.1 = fadd double %value_phi9.5.1, %value_phi9.p.6.1, !dbg !85
  %113 = fmul contract double %value_phi9.6.1, 0x3FF000001FF19E24, !dbg !80
  %114 = fadd contract double %113, %"bias::Float64", !dbg !80
  %115 = fcmp ule double %114, 1.250000e-01, !dbg !82
  %value_phi9.p.v.7.1 = select i1 %115, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.7.1 = fmul double %114, %value_phi9.p.v.7.1, !dbg !85
  %value_phi9.7.1 = fadd double %value_phi9.6.1, %value_phi9.p.7.1, !dbg !85
  %niter.next.7.1 = add i64 %niter.1, 8, !dbg !58
  %niter.ncmp.7.1 = icmp eq i64 %niter.next.7.1, %unroll_iter.1, !dbg !58
  br i1 %niter.ncmp.7.1, label %L57.loopexit.unr-lcssa.1, label %L34.1, !dbg !58

L57.loopexit.unr-lcssa.1:                         ; preds = %L34.1, %L57.loopexit
  %value_phi9.lcssa.ph.1 = phi double [ undef, %L57.loopexit ], [ %value_phi9.7.1, %L34.1 ]
  %value_phi919.unr.1 = phi double [ %value_phi13, %L57.loopexit ], [ %value_phi9.7.1, %L34.1 ]
  br i1 %lcmp.mod.not, label %L57.loopexit.1, label %L34.epil.1, !dbg !58

L34.epil.1:                                       ; preds = %L57.loopexit.unr-lcssa.1, %L34.epil.1
  %value_phi919.epil.1 = phi double [ %value_phi9.epil.1, %L34.epil.1 ], [ %value_phi919.unr.1, %L57.loopexit.unr-lcssa.1 ], !dbg !77
  %epil.iter.1 = phi i64 [ %epil.iter.next.1, %L34.epil.1 ], [ 0, %L57.loopexit.unr-lcssa.1 ]
  %116 = fmul contract double %value_phi919.epil.1, 0x3FF000001FF19E24, !dbg !80
  %117 = fadd contract double %116, %"bias::Float64", !dbg !80
  %118 = fcmp ule double %117, 1.250000e-01, !dbg !82
  %value_phi9.p.v.epil.1 = select i1 %118, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.epil.1 = fmul double %117, %value_phi9.p.v.epil.1, !dbg !85
  %value_phi9.epil.1 = fadd double %value_phi919.epil.1, %value_phi9.p.epil.1, !dbg !85
  %epil.iter.next.1 = add i64 %epil.iter.1, 1, !dbg !58
  %epil.iter.cmp.1.not = icmp eq i64 %epil.iter.next.1, %xtraiter, !dbg !58
  br i1 %epil.iter.cmp.1.not, label %L57.loopexit.1, label %L34.epil.1, !dbg !58, !llvm.loop !86

L57.loopexit.1:                                   ; preds = %L34.epil.1, %L57.loopexit.unr-lcssa.1
  %value_phi9.lcssa.1 = phi double [ %value_phi9.lcssa.ph.1, %L57.loopexit.unr-lcssa.1 ], [ %value_phi9.epil.1, %L34.epil.1 ], !dbg !85
  %119 = fmul contract double %value_phi9.lcssa.1, 0x3FF000001FF19E24, !dbg !60
  %120 = fadd contract double %119, %"bias::Float64", !dbg !60
  %121 = fcmp ule double %120, 1.250000e-01, !dbg !66
  %value_phi13.p.v.1 = select i1 %121, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.1 = fmul double %120, %value_phi13.p.v.1, !dbg !70
  %value_phi13.1 = fadd double %value_phi9.lcssa.1, %value_phi13.p.1, !dbg !70
  %122 = call double @llvm.fabs.f64(double %120), !dbg !71
  %123 = fmul contract double %122, 1.000000e-09, !dbg !76
  %124 = fadd contract double %91, %123, !dbg !76
  br i1 %58, label %L57.loopexit.unr-lcssa.2, label %L16.new.2, !dbg !58

L16.new.2:                                        ; preds = %L57.loopexit.1
  %unroll_iter.2 = and i64 %., 9223372036854775800, !dbg !58
  br label %L34.2, !dbg !58

L34.2:                                            ; preds = %L34.2, %L16.new.2
  %value_phi919.2 = phi double [ %value_phi13.1, %L16.new.2 ], [ %value_phi9.7.2, %L34.2 ], !dbg !77
  %niter.2 = phi i64 [ 0, %L16.new.2 ], [ %niter.next.7.2, %L34.2 ]
  %125 = fmul contract double %value_phi919.2, 0x3FF000001FF19E24, !dbg !80
  %126 = fadd contract double %125, %"bias::Float64", !dbg !80
  %127 = fcmp ule double %126, 1.250000e-01, !dbg !82
  %value_phi9.p.v.249 = select i1 %127, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.250 = fmul double %126, %value_phi9.p.v.249, !dbg !85
  %value_phi9.251 = fadd double %value_phi919.2, %value_phi9.p.250, !dbg !85
  %128 = fmul contract double %value_phi9.251, 0x3FF000001FF19E24, !dbg !80
  %129 = fadd contract double %128, %"bias::Float64", !dbg !80
  %130 = fcmp ule double %129, 1.250000e-01, !dbg !82
  %value_phi9.p.v.1.2 = select i1 %130, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.1.2 = fmul double %129, %value_phi9.p.v.1.2, !dbg !85
  %value_phi9.1.2 = fadd double %value_phi9.251, %value_phi9.p.1.2, !dbg !85
  %131 = fmul contract double %value_phi9.1.2, 0x3FF000001FF19E24, !dbg !80
  %132 = fadd contract double %131, %"bias::Float64", !dbg !80
  %133 = fcmp ule double %132, 1.250000e-01, !dbg !82
  %value_phi9.p.v.2.2 = select i1 %133, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.2.2 = fmul double %132, %value_phi9.p.v.2.2, !dbg !85
  %value_phi9.2.2 = fadd double %value_phi9.1.2, %value_phi9.p.2.2, !dbg !85
  %134 = fmul contract double %value_phi9.2.2, 0x3FF000001FF19E24, !dbg !80
  %135 = fadd contract double %134, %"bias::Float64", !dbg !80
  %136 = fcmp ule double %135, 1.250000e-01, !dbg !82
  %value_phi9.p.v.3.2 = select i1 %136, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.3.2 = fmul double %135, %value_phi9.p.v.3.2, !dbg !85
  %value_phi9.3.2 = fadd double %value_phi9.2.2, %value_phi9.p.3.2, !dbg !85
  %137 = fmul contract double %value_phi9.3.2, 0x3FF000001FF19E24, !dbg !80
  %138 = fadd contract double %137, %"bias::Float64", !dbg !80
  %139 = fcmp ule double %138, 1.250000e-01, !dbg !82
  %value_phi9.p.v.4.2 = select i1 %139, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.4.2 = fmul double %138, %value_phi9.p.v.4.2, !dbg !85
  %value_phi9.4.2 = fadd double %value_phi9.3.2, %value_phi9.p.4.2, !dbg !85
  %140 = fmul contract double %value_phi9.4.2, 0x3FF000001FF19E24, !dbg !80
  %141 = fadd contract double %140, %"bias::Float64", !dbg !80
  %142 = fcmp ule double %141, 1.250000e-01, !dbg !82
  %value_phi9.p.v.5.2 = select i1 %142, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.5.2 = fmul double %141, %value_phi9.p.v.5.2, !dbg !85
  %value_phi9.5.2 = fadd double %value_phi9.4.2, %value_phi9.p.5.2, !dbg !85
  %143 = fmul contract double %value_phi9.5.2, 0x3FF000001FF19E24, !dbg !80
  %144 = fadd contract double %143, %"bias::Float64", !dbg !80
  %145 = fcmp ule double %144, 1.250000e-01, !dbg !82
  %value_phi9.p.v.6.2 = select i1 %145, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.6.2 = fmul double %144, %value_phi9.p.v.6.2, !dbg !85
  %value_phi9.6.2 = fadd double %value_phi9.5.2, %value_phi9.p.6.2, !dbg !85
  %146 = fmul contract double %value_phi9.6.2, 0x3FF000001FF19E24, !dbg !80
  %147 = fadd contract double %146, %"bias::Float64", !dbg !80
  %148 = fcmp ule double %147, 1.250000e-01, !dbg !82
  %value_phi9.p.v.7.2 = select i1 %148, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.7.2 = fmul double %147, %value_phi9.p.v.7.2, !dbg !85
  %value_phi9.7.2 = fadd double %value_phi9.6.2, %value_phi9.p.7.2, !dbg !85
  %niter.next.7.2 = add i64 %niter.2, 8, !dbg !58
  %niter.ncmp.7.2 = icmp eq i64 %niter.next.7.2, %unroll_iter.2, !dbg !58
  br i1 %niter.ncmp.7.2, label %L57.loopexit.unr-lcssa.2, label %L34.2, !dbg !58

L57.loopexit.unr-lcssa.2:                         ; preds = %L34.2, %L57.loopexit.1
  %value_phi9.lcssa.ph.2 = phi double [ undef, %L57.loopexit.1 ], [ %value_phi9.7.2, %L34.2 ]
  %value_phi919.unr.2 = phi double [ %value_phi13.1, %L57.loopexit.1 ], [ %value_phi9.7.2, %L34.2 ]
  br i1 %lcmp.mod.not, label %L57.loopexit.2, label %L34.epil.2, !dbg !58

L34.epil.2:                                       ; preds = %L57.loopexit.unr-lcssa.2, %L34.epil.2
  %value_phi919.epil.2 = phi double [ %value_phi9.epil.2, %L34.epil.2 ], [ %value_phi919.unr.2, %L57.loopexit.unr-lcssa.2 ], !dbg !77
  %epil.iter.2 = phi i64 [ %epil.iter.next.2, %L34.epil.2 ], [ 0, %L57.loopexit.unr-lcssa.2 ]
  %149 = fmul contract double %value_phi919.epil.2, 0x3FF000001FF19E24, !dbg !80
  %150 = fadd contract double %149, %"bias::Float64", !dbg !80
  %151 = fcmp ule double %150, 1.250000e-01, !dbg !82
  %value_phi9.p.v.epil.2 = select i1 %151, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.epil.2 = fmul double %150, %value_phi9.p.v.epil.2, !dbg !85
  %value_phi9.epil.2 = fadd double %value_phi919.epil.2, %value_phi9.p.epil.2, !dbg !85
  %epil.iter.next.2 = add i64 %epil.iter.2, 1, !dbg !58
  %epil.iter.cmp.2.not = icmp eq i64 %epil.iter.next.2, %xtraiter, !dbg !58
  br i1 %epil.iter.cmp.2.not, label %L57.loopexit.2, label %L34.epil.2, !dbg !58, !llvm.loop !86

L57.loopexit.2:                                   ; preds = %L34.epil.2, %L57.loopexit.unr-lcssa.2
  %value_phi9.lcssa.2 = phi double [ %value_phi9.lcssa.ph.2, %L57.loopexit.unr-lcssa.2 ], [ %value_phi9.epil.2, %L34.epil.2 ], !dbg !85
  %152 = fmul contract double %value_phi9.lcssa.2, 0x3FF000001FF19E24, !dbg !60
  %153 = fadd contract double %152, %"bias::Float64", !dbg !60
  %154 = fcmp ule double %153, 1.250000e-01, !dbg !66
  %value_phi13.p.v.2 = select i1 %154, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.2 = fmul double %153, %value_phi13.p.v.2, !dbg !70
  %value_phi13.2 = fadd double %value_phi9.lcssa.2, %value_phi13.p.2, !dbg !70
  %155 = call double @llvm.fabs.f64(double %153), !dbg !71
  %156 = fmul contract double %155, 1.000000e-09, !dbg !76
  %157 = fadd contract double %124, %156, !dbg !76
  br i1 %58, label %L57.loopexit.unr-lcssa.3, label %L16.new.3, !dbg !58

L16.new.3:                                        ; preds = %L57.loopexit.2
  %unroll_iter.3 = and i64 %., 9223372036854775800, !dbg !58
  br label %L34.3, !dbg !58

L34.3:                                            ; preds = %L34.3, %L16.new.3
  %value_phi919.3 = phi double [ %value_phi13.2, %L16.new.3 ], [ %value_phi9.7.3, %L34.3 ], !dbg !77
  %niter.3 = phi i64 [ 0, %L16.new.3 ], [ %niter.next.7.3, %L34.3 ]
  %158 = fmul contract double %value_phi919.3, 0x3FF000001FF19E24, !dbg !80
  %159 = fadd contract double %158, %"bias::Float64", !dbg !80
  %160 = fcmp ule double %159, 1.250000e-01, !dbg !82
  %value_phi9.p.v.352 = select i1 %160, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.353 = fmul double %159, %value_phi9.p.v.352, !dbg !85
  %value_phi9.354 = fadd double %value_phi919.3, %value_phi9.p.353, !dbg !85
  %161 = fmul contract double %value_phi9.354, 0x3FF000001FF19E24, !dbg !80
  %162 = fadd contract double %161, %"bias::Float64", !dbg !80
  %163 = fcmp ule double %162, 1.250000e-01, !dbg !82
  %value_phi9.p.v.1.3 = select i1 %163, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.1.3 = fmul double %162, %value_phi9.p.v.1.3, !dbg !85
  %value_phi9.1.3 = fadd double %value_phi9.354, %value_phi9.p.1.3, !dbg !85
  %164 = fmul contract double %value_phi9.1.3, 0x3FF000001FF19E24, !dbg !80
  %165 = fadd contract double %164, %"bias::Float64", !dbg !80
  %166 = fcmp ule double %165, 1.250000e-01, !dbg !82
  %value_phi9.p.v.2.3 = select i1 %166, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.2.3 = fmul double %165, %value_phi9.p.v.2.3, !dbg !85
  %value_phi9.2.3 = fadd double %value_phi9.1.3, %value_phi9.p.2.3, !dbg !85
  %167 = fmul contract double %value_phi9.2.3, 0x3FF000001FF19E24, !dbg !80
  %168 = fadd contract double %167, %"bias::Float64", !dbg !80
  %169 = fcmp ule double %168, 1.250000e-01, !dbg !82
  %value_phi9.p.v.3.3 = select i1 %169, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.3.3 = fmul double %168, %value_phi9.p.v.3.3, !dbg !85
  %value_phi9.3.3 = fadd double %value_phi9.2.3, %value_phi9.p.3.3, !dbg !85
  %170 = fmul contract double %value_phi9.3.3, 0x3FF000001FF19E24, !dbg !80
  %171 = fadd contract double %170, %"bias::Float64", !dbg !80
  %172 = fcmp ule double %171, 1.250000e-01, !dbg !82
  %value_phi9.p.v.4.3 = select i1 %172, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.4.3 = fmul double %171, %value_phi9.p.v.4.3, !dbg !85
  %value_phi9.4.3 = fadd double %value_phi9.3.3, %value_phi9.p.4.3, !dbg !85
  %173 = fmul contract double %value_phi9.4.3, 0x3FF000001FF19E24, !dbg !80
  %174 = fadd contract double %173, %"bias::Float64", !dbg !80
  %175 = fcmp ule double %174, 1.250000e-01, !dbg !82
  %value_phi9.p.v.5.3 = select i1 %175, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.5.3 = fmul double %174, %value_phi9.p.v.5.3, !dbg !85
  %value_phi9.5.3 = fadd double %value_phi9.4.3, %value_phi9.p.5.3, !dbg !85
  %176 = fmul contract double %value_phi9.5.3, 0x3FF000001FF19E24, !dbg !80
  %177 = fadd contract double %176, %"bias::Float64", !dbg !80
  %178 = fcmp ule double %177, 1.250000e-01, !dbg !82
  %value_phi9.p.v.6.3 = select i1 %178, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.6.3 = fmul double %177, %value_phi9.p.v.6.3, !dbg !85
  %value_phi9.6.3 = fadd double %value_phi9.5.3, %value_phi9.p.6.3, !dbg !85
  %179 = fmul contract double %value_phi9.6.3, 0x3FF000001FF19E24, !dbg !80
  %180 = fadd contract double %179, %"bias::Float64", !dbg !80
  %181 = fcmp ule double %180, 1.250000e-01, !dbg !82
  %value_phi9.p.v.7.3 = select i1 %181, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.7.3 = fmul double %180, %value_phi9.p.v.7.3, !dbg !85
  %value_phi9.7.3 = fadd double %value_phi9.6.3, %value_phi9.p.7.3, !dbg !85
  %niter.next.7.3 = add i64 %niter.3, 8, !dbg !58
  %niter.ncmp.7.3 = icmp eq i64 %niter.next.7.3, %unroll_iter.3, !dbg !58
  br i1 %niter.ncmp.7.3, label %L57.loopexit.unr-lcssa.3, label %L34.3, !dbg !58

L57.loopexit.unr-lcssa.3:                         ; preds = %L34.3, %L57.loopexit.2
  %value_phi9.lcssa.ph.3 = phi double [ undef, %L57.loopexit.2 ], [ %value_phi9.7.3, %L34.3 ]
  %value_phi919.unr.3 = phi double [ %value_phi13.2, %L57.loopexit.2 ], [ %value_phi9.7.3, %L34.3 ]
  br i1 %lcmp.mod.not, label %L57.loopexit.3, label %L34.epil.3, !dbg !58

L34.epil.3:                                       ; preds = %L57.loopexit.unr-lcssa.3, %L34.epil.3
  %value_phi919.epil.3 = phi double [ %value_phi9.epil.3, %L34.epil.3 ], [ %value_phi919.unr.3, %L57.loopexit.unr-lcssa.3 ], !dbg !77
  %epil.iter.3 = phi i64 [ %epil.iter.next.3, %L34.epil.3 ], [ 0, %L57.loopexit.unr-lcssa.3 ]
  %182 = fmul contract double %value_phi919.epil.3, 0x3FF000001FF19E24, !dbg !80
  %183 = fadd contract double %182, %"bias::Float64", !dbg !80
  %184 = fcmp ule double %183, 1.250000e-01, !dbg !82
  %value_phi9.p.v.epil.3 = select i1 %184, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.epil.3 = fmul double %183, %value_phi9.p.v.epil.3, !dbg !85
  %value_phi9.epil.3 = fadd double %value_phi919.epil.3, %value_phi9.p.epil.3, !dbg !85
  %epil.iter.next.3 = add i64 %epil.iter.3, 1, !dbg !58
  %epil.iter.cmp.3.not = icmp eq i64 %epil.iter.next.3, %xtraiter, !dbg !58
  br i1 %epil.iter.cmp.3.not, label %L57.loopexit.3, label %L34.epil.3, !dbg !58, !llvm.loop !86

L57.loopexit.3:                                   ; preds = %L34.epil.3, %L57.loopexit.unr-lcssa.3
  %value_phi9.lcssa.3 = phi double [ %value_phi9.lcssa.ph.3, %L57.loopexit.unr-lcssa.3 ], [ %value_phi9.epil.3, %L34.epil.3 ], !dbg !85
  %185 = fmul contract double %value_phi9.lcssa.3, 0x3FF000001FF19E24, !dbg !60
  %186 = fadd contract double %185, %"bias::Float64", !dbg !60
  %187 = fcmp ule double %186, 1.250000e-01, !dbg !66
  %value_phi13.p.v.3 = select i1 %187, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.3 = fmul double %186, %value_phi13.p.v.3, !dbg !70
  %value_phi13.3 = fadd double %value_phi9.lcssa.3, %value_phi13.p.3, !dbg !70
  %188 = call double @llvm.fabs.f64(double %186), !dbg !71
  %189 = fmul contract double %188, 1.000000e-09, !dbg !76
  %190 = fadd contract double %157, %189, !dbg !76
  %niter45.next.3 = add i64 %niter45, 4, !dbg !59
  %niter45.ncmp.3 = icmp eq i64 %niter45.next.3, %unroll_iter44, !dbg !59
  br i1 %niter45.ncmp.3, label %L81.loopexit.loopexit30.unr-lcssa, label %L16, !dbg !59

L81.loopexit.loopexit.unr-lcssa:                  ; preds = %L16.us, %L16.us.preheader
  %value_phi13.us.lcssa.ph = phi double [ undef, %L16.us.preheader ], [ %value_phi13.us.7, %L16.us ]
  %.lcssa.ph = phi double [ undef, %L16.us.preheader ], [ %57, %L16.us ]
  %value_phi1321.us.unr = phi double [ %.promoted20, %L16.us.preheader ], [ %value_phi13.us.7, %L16.us ]
  %value_phi4.us.unr = phi double [ 0.000000e+00, %L16.us.preheader ], [ %57, %L16.us ]
  %lcmp.mod57.not = icmp eq i64 %xtraiter55, 0, !dbg !59
  br i1 %lcmp.mod57.not, label %L81.loopexit, label %L16.us.epil, !dbg !59

L16.us.epil:                                      ; preds = %L81.loopexit.loopexit.unr-lcssa, %L16.us.epil
  %value_phi1321.us.epil = phi double [ %value_phi13.us.epil, %L16.us.epil ], [ %value_phi1321.us.unr, %L81.loopexit.loopexit.unr-lcssa ]
  %value_phi4.us.epil = phi double [ %196, %L16.us.epil ], [ %value_phi4.us.unr, %L81.loopexit.loopexit.unr-lcssa ]
  %epil.iter56 = phi i64 [ %epil.iter56.next, %L16.us.epil ], [ 0, %L81.loopexit.loopexit.unr-lcssa ]
  %191 = fmul contract double %value_phi1321.us.epil, 0x3FF000001FF19E24, !dbg !60
  %192 = fadd contract double %191, %"bias::Float64", !dbg !60
  %193 = fcmp ule double %192, 1.250000e-01, !dbg !66
  %value_phi13.p.v.us.epil = select i1 %193, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.us.epil = fmul double %192, %value_phi13.p.v.us.epil, !dbg !70
  %value_phi13.us.epil = fadd double %value_phi1321.us.epil, %value_phi13.p.us.epil, !dbg !70
  %194 = call double @llvm.fabs.f64(double %192), !dbg !71
  %195 = fmul contract double %194, 1.000000e-09, !dbg !76
  %196 = fadd contract double %value_phi4.us.epil, %195, !dbg !76
  %epil.iter56.next = add i64 %epil.iter56, 1, !dbg !59
  %epil.iter56.cmp.not = icmp eq i64 %epil.iter56.next, %xtraiter55, !dbg !59
  br i1 %epil.iter56.cmp.not, label %L81.loopexit, label %L16.us.epil, !dbg !59, !llvm.loop !88

L81.loopexit.loopexit30.unr-lcssa:                ; preds = %L57.loopexit.3, %L16.preheader29
  %value_phi13.lcssa.ph = phi double [ undef, %L16.preheader29 ], [ %value_phi13.3, %L57.loopexit.3 ]
  %.lcssa31.ph = phi double [ undef, %L16.preheader29 ], [ %190, %L57.loopexit.3 ]
  %value_phi1321.unr = phi double [ %.promoted20, %L16.preheader29 ], [ %value_phi13.3, %L57.loopexit.3 ]
  %value_phi4.unr = phi double [ 0.000000e+00, %L16.preheader29 ], [ %190, %L57.loopexit.3 ]
  %lcmp.mod41.not = icmp eq i64 %xtraiter33, 0, !dbg !58
  br i1 %lcmp.mod41.not, label %L81.loopexit, label %L16.epil, !dbg !58

L16.epil:                                         ; preds = %L81.loopexit.loopexit30.unr-lcssa, %L57.loopexit.epil
  %value_phi1321.epil = phi double [ %value_phi13.epil, %L57.loopexit.epil ], [ %value_phi1321.unr, %L81.loopexit.loopexit30.unr-lcssa ]
  %value_phi4.epil = phi double [ %230, %L57.loopexit.epil ], [ %value_phi4.unr, %L81.loopexit.loopexit30.unr-lcssa ]
  %epil.iter40 = phi i64 [ %epil.iter40.next, %L57.loopexit.epil ], [ 0, %L81.loopexit.loopexit30.unr-lcssa ]
  %xtraiter.epil = and i64 %., 7, !dbg !58
  %197 = icmp ult i64 %5, 7, !dbg !58
  br i1 %197, label %L57.loopexit.unr-lcssa.epil, label %L16.new.epil, !dbg !58

L16.new.epil:                                     ; preds = %L16.epil
  %unroll_iter.epil = and i64 %., 9223372036854775800, !dbg !58
  br label %L34.epil34, !dbg !58

L34.epil34:                                       ; preds = %L34.epil34, %L16.new.epil
  %value_phi919.epil35 = phi double [ %value_phi1321.epil, %L16.new.epil ], [ %value_phi9.7.epil, %L34.epil34 ], !dbg !77
  %niter.epil = phi i64 [ 0, %L16.new.epil ], [ %niter.next.7.epil, %L34.epil34 ]
  %198 = fmul contract double %value_phi919.epil35, 0x3FF000001FF19E24, !dbg !80
  %199 = fadd contract double %198, %"bias::Float64", !dbg !80
  %200 = fcmp ule double %199, 1.250000e-01, !dbg !82
  %value_phi9.p.v.epil37 = select i1 %200, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.epil38 = fmul double %199, %value_phi9.p.v.epil37, !dbg !85
  %value_phi9.epil39 = fadd double %value_phi919.epil35, %value_phi9.p.epil38, !dbg !85
  %201 = fmul contract double %value_phi9.epil39, 0x3FF000001FF19E24, !dbg !80
  %202 = fadd contract double %201, %"bias::Float64", !dbg !80
  %203 = fcmp ule double %202, 1.250000e-01, !dbg !82
  %value_phi9.p.v.1.epil = select i1 %203, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.1.epil = fmul double %202, %value_phi9.p.v.1.epil, !dbg !85
  %value_phi9.1.epil = fadd double %value_phi9.epil39, %value_phi9.p.1.epil, !dbg !85
  %204 = fmul contract double %value_phi9.1.epil, 0x3FF000001FF19E24, !dbg !80
  %205 = fadd contract double %204, %"bias::Float64", !dbg !80
  %206 = fcmp ule double %205, 1.250000e-01, !dbg !82
  %value_phi9.p.v.2.epil = select i1 %206, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.2.epil = fmul double %205, %value_phi9.p.v.2.epil, !dbg !85
  %value_phi9.2.epil = fadd double %value_phi9.1.epil, %value_phi9.p.2.epil, !dbg !85
  %207 = fmul contract double %value_phi9.2.epil, 0x3FF000001FF19E24, !dbg !80
  %208 = fadd contract double %207, %"bias::Float64", !dbg !80
  %209 = fcmp ule double %208, 1.250000e-01, !dbg !82
  %value_phi9.p.v.3.epil = select i1 %209, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.3.epil = fmul double %208, %value_phi9.p.v.3.epil, !dbg !85
  %value_phi9.3.epil = fadd double %value_phi9.2.epil, %value_phi9.p.3.epil, !dbg !85
  %210 = fmul contract double %value_phi9.3.epil, 0x3FF000001FF19E24, !dbg !80
  %211 = fadd contract double %210, %"bias::Float64", !dbg !80
  %212 = fcmp ule double %211, 1.250000e-01, !dbg !82
  %value_phi9.p.v.4.epil = select i1 %212, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.4.epil = fmul double %211, %value_phi9.p.v.4.epil, !dbg !85
  %value_phi9.4.epil = fadd double %value_phi9.3.epil, %value_phi9.p.4.epil, !dbg !85
  %213 = fmul contract double %value_phi9.4.epil, 0x3FF000001FF19E24, !dbg !80
  %214 = fadd contract double %213, %"bias::Float64", !dbg !80
  %215 = fcmp ule double %214, 1.250000e-01, !dbg !82
  %value_phi9.p.v.5.epil = select i1 %215, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.5.epil = fmul double %214, %value_phi9.p.v.5.epil, !dbg !85
  %value_phi9.5.epil = fadd double %value_phi9.4.epil, %value_phi9.p.5.epil, !dbg !85
  %216 = fmul contract double %value_phi9.5.epil, 0x3FF000001FF19E24, !dbg !80
  %217 = fadd contract double %216, %"bias::Float64", !dbg !80
  %218 = fcmp ule double %217, 1.250000e-01, !dbg !82
  %value_phi9.p.v.6.epil = select i1 %218, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.6.epil = fmul double %217, %value_phi9.p.v.6.epil, !dbg !85
  %value_phi9.6.epil = fadd double %value_phi9.5.epil, %value_phi9.p.6.epil, !dbg !85
  %219 = fmul contract double %value_phi9.6.epil, 0x3FF000001FF19E24, !dbg !80
  %220 = fadd contract double %219, %"bias::Float64", !dbg !80
  %221 = fcmp ule double %220, 1.250000e-01, !dbg !82
  %value_phi9.p.v.7.epil = select i1 %221, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.7.epil = fmul double %220, %value_phi9.p.v.7.epil, !dbg !85
  %value_phi9.7.epil = fadd double %value_phi9.6.epil, %value_phi9.p.7.epil, !dbg !85
  %niter.next.7.epil = add i64 %niter.epil, 8, !dbg !58
  %niter.ncmp.7.epil = icmp eq i64 %niter.next.7.epil, %unroll_iter.epil, !dbg !58
  br i1 %niter.ncmp.7.epil, label %L57.loopexit.unr-lcssa.epil, label %L34.epil34, !dbg !58

L57.loopexit.unr-lcssa.epil:                      ; preds = %L34.epil34, %L16.epil
  %value_phi9.lcssa.ph.epil = phi double [ undef, %L16.epil ], [ %value_phi9.7.epil, %L34.epil34 ]
  %value_phi919.unr.epil = phi double [ %value_phi1321.epil, %L16.epil ], [ %value_phi9.7.epil, %L34.epil34 ]
  %lcmp.mod.epil.not = icmp eq i64 %xtraiter.epil, 0, !dbg !58
  br i1 %lcmp.mod.epil.not, label %L57.loopexit.epil, label %L34.epil.epil, !dbg !58

L34.epil.epil:                                    ; preds = %L57.loopexit.unr-lcssa.epil, %L34.epil.epil
  %value_phi919.epil.epil = phi double [ %value_phi9.epil.epil, %L34.epil.epil ], [ %value_phi919.unr.epil, %L57.loopexit.unr-lcssa.epil ], !dbg !77
  %epil.iter.epil = phi i64 [ %epil.iter.next.epil, %L34.epil.epil ], [ 0, %L57.loopexit.unr-lcssa.epil ]
  %222 = fmul contract double %value_phi919.epil.epil, 0x3FF000001FF19E24, !dbg !80
  %223 = fadd contract double %222, %"bias::Float64", !dbg !80
  %224 = fcmp ule double %223, 1.250000e-01, !dbg !82
  %value_phi9.p.v.epil.epil = select i1 %224, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !85
  %value_phi9.p.epil.epil = fmul double %223, %value_phi9.p.v.epil.epil, !dbg !85
  %value_phi9.epil.epil = fadd double %value_phi919.epil.epil, %value_phi9.p.epil.epil, !dbg !85
  %epil.iter.next.epil = add i64 %epil.iter.epil, 1, !dbg !58
  %epil.iter.cmp.epil.not = icmp eq i64 %epil.iter.next.epil, %xtraiter.epil, !dbg !58
  br i1 %epil.iter.cmp.epil.not, label %L57.loopexit.epil, label %L34.epil.epil, !dbg !58, !llvm.loop !86

L57.loopexit.epil:                                ; preds = %L34.epil.epil, %L57.loopexit.unr-lcssa.epil
  %value_phi9.lcssa.epil = phi double [ %value_phi9.lcssa.ph.epil, %L57.loopexit.unr-lcssa.epil ], [ %value_phi9.epil.epil, %L34.epil.epil ], !dbg !85
  %225 = fmul contract double %value_phi9.lcssa.epil, 0x3FF000001FF19E24, !dbg !60
  %226 = fadd contract double %225, %"bias::Float64", !dbg !60
  %227 = fcmp ule double %226, 1.250000e-01, !dbg !66
  %value_phi13.p.v.epil = select i1 %227, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !70
  %value_phi13.p.epil = fmul double %226, %value_phi13.p.v.epil, !dbg !70
  %value_phi13.epil = fadd double %value_phi9.lcssa.epil, %value_phi13.p.epil, !dbg !70
  %228 = call double @llvm.fabs.f64(double %226), !dbg !71
  %229 = fmul contract double %228, 1.000000e-09, !dbg !76
  %230 = fadd contract double %value_phi4.epil, %229, !dbg !76
  %epil.iter40.next = add i64 %epil.iter40, 1, !dbg !59
  %epil.iter40.cmp.not = icmp eq i64 %epil.iter40.next, %xtraiter33, !dbg !59
  br i1 %epil.iter40.cmp.not, label %L81.loopexit, label %L16.epil, !dbg !59, !llvm.loop !89

L81.loopexit:                                     ; preds = %L81.loopexit.loopexit30.unr-lcssa, %L57.loopexit.epil, %L81.loopexit.loopexit.unr-lcssa, %L16.us.epil
  %.us-phi = phi double [ %value_phi13.us.lcssa.ph, %L81.loopexit.loopexit.unr-lcssa ], [ %value_phi13.us.epil, %L16.us.epil ], [ %value_phi13.lcssa.ph, %L81.loopexit.loopexit30.unr-lcssa ], [ %value_phi13.epil, %L57.loopexit.epil ]
  %.us-phi23 = phi double [ %.lcssa.ph, %L81.loopexit.loopexit.unr-lcssa ], [ %196, %L16.us.epil ], [ %.lcssa31.ph, %L81.loopexit.loopexit30.unr-lcssa ], [ %230, %L57.loopexit.epil ]
  store double %.us-phi, ptr %"box::Box", align 8, !tbaa !45, !alias.scope !49, !noalias !52
  br label %L81, !dbg !41

L81:                                              ; preds = %L81.loopexit, %top.L81_crit_edge
  %"box::Box.x17" = phi double [ %"box::Box.x17.pre", %top.L81_crit_edge ], [ %.us-phi, %L81.loopexit ], !dbg !41
  %value_phi16 = phi double [ 0.000000e+00, %top.L81_crit_edge ], [ %.us-phi23, %L81.loopexit ]
  %231 = fadd double %value_phi16, %"box::Box.x17", !dbg !90
  ret double %231, !dbg !44
}
