; Function Signature: blocked_capture(Int64, Int64, Float64, Float64)
; Function Attrs: uwtable
define swiftcc double @julia_blocked_capture_654(ptr nonnull swiftself %pgcstack, i64 signext %"nblocks::Int64", i64 signext %"period::Int64", double %"x::Float64", double %"bias::Float64") #0 !dbg !4 {
top:
  call void @llvm.dbg.value(metadata i64 %"nblocks::Int64", metadata !14, metadata !DIExpression()), !dbg !18
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
  %".nblocks::Int64" = call i64 @llvm.smax.i64(i64 %"nblocks::Int64", i64 0), !dbg !25
  %2 = icmp slt i64 %"nblocks::Int64", 1, !dbg !26
  br i1 %2, label %L80, label %L16.preheader, !dbg !38

L16.preheader:                                    ; preds = %top
  %3 = add i64 %"period::Int64", -1
  %. = call i64 @llvm.smax.i64(i64 %3, i64 0)
  %4 = icmp slt i64 %3, 1
  br i1 %4, label %L16.us.preheader, label %L16.preheader26, !dbg !39

L16.preheader26:                                  ; preds = %L16.preheader
  %5 = add nsw i64 %., -1, !dbg !40
  %6 = add nsw i64 %".nblocks::Int64", -1, !dbg !40
  %xtraiter30 = and i64 %".nblocks::Int64", 3, !dbg !40
  %7 = icmp ult i64 %6, 3, !dbg !40
  br i1 %7, label %L80.loopexit27.unr-lcssa, label %L16.preheader26.new, !dbg !40

L16.preheader26.new:                              ; preds = %L16.preheader26
  %unroll_iter41 = and i64 %".nblocks::Int64", 9223372036854775804, !dbg !40
  br label %L16, !dbg !40

L16.us.preheader:                                 ; preds = %L16.preheader
  %8 = add nsw i64 %".nblocks::Int64", -1, !dbg !41
  %xtraiter52 = and i64 %".nblocks::Int64", 7, !dbg !41
  %9 = icmp ult i64 %8, 7, !dbg !41
  br i1 %9, label %L80.loopexit.unr-lcssa, label %L16.us.preheader.new, !dbg !41

L16.us.preheader.new:                             ; preds = %L16.us.preheader
  %unroll_iter57 = and i64 %".nblocks::Int64", 9223372036854775800, !dbg !41
  br label %L16.us, !dbg !41

L16.us:                                           ; preds = %L16.us, %L16.us.preheader.new
  %value_phi4.us = phi double [ 0.000000e+00, %L16.us.preheader.new ], [ %57, %L16.us ]
  %value_phi5.us = phi double [ %"x::Float64", %L16.us.preheader.new ], [ %value_phi15.us.7, %L16.us ]
  %niter58 = phi i64 [ 0, %L16.us.preheader.new ], [ %niter58.next.7, %L16.us ]
  %10 = fmul contract double %value_phi5.us, 0x3FF000001FF19E24, !dbg !42
  %11 = fadd contract double %10, %"bias::Float64", !dbg !42
  %12 = fcmp ule double %11, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us = select i1 %12, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us = fmul double %11, %value_phi15.p.v.us, !dbg !52
  %value_phi15.us = fadd double %value_phi5.us, %value_phi15.p.us, !dbg !52
  %13 = call double @llvm.fabs.f64(double %11), !dbg !53
  %14 = fmul contract double %13, 1.000000e-09, !dbg !58
  %15 = fadd contract double %value_phi4.us, %14, !dbg !58
  %16 = fmul contract double %value_phi15.us, 0x3FF000001FF19E24, !dbg !42
  %17 = fadd contract double %16, %"bias::Float64", !dbg !42
  %18 = fcmp ule double %17, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.1 = select i1 %18, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.1 = fmul double %17, %value_phi15.p.v.us.1, !dbg !52
  %value_phi15.us.1 = fadd double %value_phi15.us, %value_phi15.p.us.1, !dbg !52
  %19 = call double @llvm.fabs.f64(double %17), !dbg !53
  %20 = fmul contract double %19, 1.000000e-09, !dbg !58
  %21 = fadd contract double %15, %20, !dbg !58
  %22 = fmul contract double %value_phi15.us.1, 0x3FF000001FF19E24, !dbg !42
  %23 = fadd contract double %22, %"bias::Float64", !dbg !42
  %24 = fcmp ule double %23, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.2 = select i1 %24, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.2 = fmul double %23, %value_phi15.p.v.us.2, !dbg !52
  %value_phi15.us.2 = fadd double %value_phi15.us.1, %value_phi15.p.us.2, !dbg !52
  %25 = call double @llvm.fabs.f64(double %23), !dbg !53
  %26 = fmul contract double %25, 1.000000e-09, !dbg !58
  %27 = fadd contract double %21, %26, !dbg !58
  %28 = fmul contract double %value_phi15.us.2, 0x3FF000001FF19E24, !dbg !42
  %29 = fadd contract double %28, %"bias::Float64", !dbg !42
  %30 = fcmp ule double %29, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.3 = select i1 %30, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.3 = fmul double %29, %value_phi15.p.v.us.3, !dbg !52
  %value_phi15.us.3 = fadd double %value_phi15.us.2, %value_phi15.p.us.3, !dbg !52
  %31 = call double @llvm.fabs.f64(double %29), !dbg !53
  %32 = fmul contract double %31, 1.000000e-09, !dbg !58
  %33 = fadd contract double %27, %32, !dbg !58
  %34 = fmul contract double %value_phi15.us.3, 0x3FF000001FF19E24, !dbg !42
  %35 = fadd contract double %34, %"bias::Float64", !dbg !42
  %36 = fcmp ule double %35, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.4 = select i1 %36, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.4 = fmul double %35, %value_phi15.p.v.us.4, !dbg !52
  %value_phi15.us.4 = fadd double %value_phi15.us.3, %value_phi15.p.us.4, !dbg !52
  %37 = call double @llvm.fabs.f64(double %35), !dbg !53
  %38 = fmul contract double %37, 1.000000e-09, !dbg !58
  %39 = fadd contract double %33, %38, !dbg !58
  %40 = fmul contract double %value_phi15.us.4, 0x3FF000001FF19E24, !dbg !42
  %41 = fadd contract double %40, %"bias::Float64", !dbg !42
  %42 = fcmp ule double %41, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.5 = select i1 %42, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.5 = fmul double %41, %value_phi15.p.v.us.5, !dbg !52
  %value_phi15.us.5 = fadd double %value_phi15.us.4, %value_phi15.p.us.5, !dbg !52
  %43 = call double @llvm.fabs.f64(double %41), !dbg !53
  %44 = fmul contract double %43, 1.000000e-09, !dbg !58
  %45 = fadd contract double %39, %44, !dbg !58
  %46 = fmul contract double %value_phi15.us.5, 0x3FF000001FF19E24, !dbg !42
  %47 = fadd contract double %46, %"bias::Float64", !dbg !42
  %48 = fcmp ule double %47, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.6 = select i1 %48, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.6 = fmul double %47, %value_phi15.p.v.us.6, !dbg !52
  %value_phi15.us.6 = fadd double %value_phi15.us.5, %value_phi15.p.us.6, !dbg !52
  %49 = call double @llvm.fabs.f64(double %47), !dbg !53
  %50 = fmul contract double %49, 1.000000e-09, !dbg !58
  %51 = fadd contract double %45, %50, !dbg !58
  %52 = fmul contract double %value_phi15.us.6, 0x3FF000001FF19E24, !dbg !42
  %53 = fadd contract double %52, %"bias::Float64", !dbg !42
  %54 = fcmp ule double %53, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.7 = select i1 %54, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.7 = fmul double %53, %value_phi15.p.v.us.7, !dbg !52
  %value_phi15.us.7 = fadd double %value_phi15.us.6, %value_phi15.p.us.7, !dbg !52
  %55 = call double @llvm.fabs.f64(double %53), !dbg !53
  %56 = fmul contract double %55, 1.000000e-09, !dbg !58
  %57 = fadd contract double %51, %56, !dbg !58
  %niter58.next.7 = add i64 %niter58, 8, !dbg !41
  %niter58.ncmp.7 = icmp eq i64 %niter58.next.7, %unroll_iter57, !dbg !41
  br i1 %niter58.ncmp.7, label %L80.loopexit.unr-lcssa, label %L16.us, !dbg !41

L16:                                              ; preds = %L57.loopexit.3, %L16.preheader26.new
  %value_phi4 = phi double [ 0.000000e+00, %L16.preheader26.new ], [ %190, %L57.loopexit.3 ]
  %value_phi5 = phi double [ %"x::Float64", %L16.preheader26.new ], [ %value_phi15.3, %L57.loopexit.3 ]
  %niter42 = phi i64 [ 0, %L16.preheader26.new ], [ %niter42.next.3, %L57.loopexit.3 ]
  %xtraiter = and i64 %., 7, !dbg !40
  %58 = icmp ult i64 %5, 7, !dbg !40
  br i1 %58, label %L57.loopexit.unr-lcssa, label %L16.new, !dbg !40

L16.new:                                          ; preds = %L16
  %unroll_iter = and i64 %., 9223372036854775800, !dbg !40
  br label %L35, !dbg !40

L35:                                              ; preds = %L35, %L16.new
  %value_phi10 = phi double [ %value_phi5, %L16.new ], [ %value_phi11.7, %L35 ]
  %niter = phi i64 [ 0, %L16.new ], [ %niter.next.7, %L35 ]
  %59 = fmul contract double %value_phi10, 0x3FF000001FF19E24, !dbg !59
  %60 = fadd contract double %59, %"bias::Float64", !dbg !59
  %61 = fcmp ule double %60, 1.250000e-01, !dbg !62
  %value_phi11.p.v = select i1 %61, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p = fmul double %60, %value_phi11.p.v, !dbg !65
  %value_phi11 = fadd double %value_phi10, %value_phi11.p, !dbg !65
  %62 = fmul contract double %value_phi11, 0x3FF000001FF19E24, !dbg !59
  %63 = fadd contract double %62, %"bias::Float64", !dbg !59
  %64 = fcmp ule double %63, 1.250000e-01, !dbg !62
  %value_phi11.p.v.1 = select i1 %64, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.1 = fmul double %63, %value_phi11.p.v.1, !dbg !65
  %value_phi11.1 = fadd double %value_phi11, %value_phi11.p.1, !dbg !65
  %65 = fmul contract double %value_phi11.1, 0x3FF000001FF19E24, !dbg !59
  %66 = fadd contract double %65, %"bias::Float64", !dbg !59
  %67 = fcmp ule double %66, 1.250000e-01, !dbg !62
  %value_phi11.p.v.2 = select i1 %67, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.2 = fmul double %66, %value_phi11.p.v.2, !dbg !65
  %value_phi11.2 = fadd double %value_phi11.1, %value_phi11.p.2, !dbg !65
  %68 = fmul contract double %value_phi11.2, 0x3FF000001FF19E24, !dbg !59
  %69 = fadd contract double %68, %"bias::Float64", !dbg !59
  %70 = fcmp ule double %69, 1.250000e-01, !dbg !62
  %value_phi11.p.v.3 = select i1 %70, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.3 = fmul double %69, %value_phi11.p.v.3, !dbg !65
  %value_phi11.3 = fadd double %value_phi11.2, %value_phi11.p.3, !dbg !65
  %71 = fmul contract double %value_phi11.3, 0x3FF000001FF19E24, !dbg !59
  %72 = fadd contract double %71, %"bias::Float64", !dbg !59
  %73 = fcmp ule double %72, 1.250000e-01, !dbg !62
  %value_phi11.p.v.4 = select i1 %73, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.4 = fmul double %72, %value_phi11.p.v.4, !dbg !65
  %value_phi11.4 = fadd double %value_phi11.3, %value_phi11.p.4, !dbg !65
  %74 = fmul contract double %value_phi11.4, 0x3FF000001FF19E24, !dbg !59
  %75 = fadd contract double %74, %"bias::Float64", !dbg !59
  %76 = fcmp ule double %75, 1.250000e-01, !dbg !62
  %value_phi11.p.v.5 = select i1 %76, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.5 = fmul double %75, %value_phi11.p.v.5, !dbg !65
  %value_phi11.5 = fadd double %value_phi11.4, %value_phi11.p.5, !dbg !65
  %77 = fmul contract double %value_phi11.5, 0x3FF000001FF19E24, !dbg !59
  %78 = fadd contract double %77, %"bias::Float64", !dbg !59
  %79 = fcmp ule double %78, 1.250000e-01, !dbg !62
  %value_phi11.p.v.6 = select i1 %79, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.6 = fmul double %78, %value_phi11.p.v.6, !dbg !65
  %value_phi11.6 = fadd double %value_phi11.5, %value_phi11.p.6, !dbg !65
  %80 = fmul contract double %value_phi11.6, 0x3FF000001FF19E24, !dbg !59
  %81 = fadd contract double %80, %"bias::Float64", !dbg !59
  %82 = fcmp ule double %81, 1.250000e-01, !dbg !62
  %value_phi11.p.v.7 = select i1 %82, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.7 = fmul double %81, %value_phi11.p.v.7, !dbg !65
  %value_phi11.7 = fadd double %value_phi11.6, %value_phi11.p.7, !dbg !65
  %niter.next.7 = add i64 %niter, 8, !dbg !40
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter, !dbg !40
  br i1 %niter.ncmp.7, label %L57.loopexit.unr-lcssa, label %L35, !dbg !40

L57.loopexit.unr-lcssa:                           ; preds = %L35, %L16
  %value_phi11.lcssa.ph = phi double [ undef, %L16 ], [ %value_phi11.7, %L35 ]
  %value_phi10.unr = phi double [ %value_phi5, %L16 ], [ %value_phi11.7, %L35 ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !40
  br i1 %lcmp.mod.not, label %L57.loopexit, label %L35.epil, !dbg !40

L35.epil:                                         ; preds = %L57.loopexit.unr-lcssa, %L35.epil
  %value_phi10.epil = phi double [ %value_phi11.epil, %L35.epil ], [ %value_phi10.unr, %L57.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %L35.epil ], [ 0, %L57.loopexit.unr-lcssa ]
  %83 = fmul contract double %value_phi10.epil, 0x3FF000001FF19E24, !dbg !59
  %84 = fadd contract double %83, %"bias::Float64", !dbg !59
  %85 = fcmp ule double %84, 1.250000e-01, !dbg !62
  %value_phi11.p.v.epil = select i1 %85, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.epil = fmul double %84, %value_phi11.p.v.epil, !dbg !65
  %value_phi11.epil = fadd double %value_phi10.epil, %value_phi11.p.epil, !dbg !65
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !40
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !40
  br i1 %epil.iter.cmp.not, label %L57.loopexit, label %L35.epil, !dbg !40, !llvm.loop !66

L57.loopexit:                                     ; preds = %L35.epil, %L57.loopexit.unr-lcssa
  %value_phi11.lcssa = phi double [ %value_phi11.lcssa.ph, %L57.loopexit.unr-lcssa ], [ %value_phi11.epil, %L35.epil ], !dbg !65
  %86 = fmul contract double %value_phi11.lcssa, 0x3FF000001FF19E24, !dbg !42
  %87 = fadd contract double %86, %"bias::Float64", !dbg !42
  %88 = fcmp ule double %87, 1.250000e-01, !dbg !48
  %value_phi15.p.v = select i1 %88, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p = fmul double %87, %value_phi15.p.v, !dbg !52
  %value_phi15 = fadd double %value_phi11.lcssa, %value_phi15.p, !dbg !52
  %89 = call double @llvm.fabs.f64(double %87), !dbg !53
  %90 = fmul contract double %89, 1.000000e-09, !dbg !58
  %91 = fadd contract double %value_phi4, %90, !dbg !58
  br i1 %58, label %L57.loopexit.unr-lcssa.1, label %L16.new.1, !dbg !40

L16.new.1:                                        ; preds = %L57.loopexit
  %unroll_iter.1 = and i64 %., 9223372036854775800, !dbg !40
  br label %L35.1, !dbg !40

L35.1:                                            ; preds = %L35.1, %L16.new.1
  %value_phi10.1 = phi double [ %value_phi15, %L16.new.1 ], [ %value_phi11.7.1, %L35.1 ]
  %niter.1 = phi i64 [ 0, %L16.new.1 ], [ %niter.next.7.1, %L35.1 ]
  %92 = fmul contract double %value_phi10.1, 0x3FF000001FF19E24, !dbg !59
  %93 = fadd contract double %92, %"bias::Float64", !dbg !59
  %94 = fcmp ule double %93, 1.250000e-01, !dbg !62
  %value_phi11.p.v.143 = select i1 %94, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.144 = fmul double %93, %value_phi11.p.v.143, !dbg !65
  %value_phi11.145 = fadd double %value_phi10.1, %value_phi11.p.144, !dbg !65
  %95 = fmul contract double %value_phi11.145, 0x3FF000001FF19E24, !dbg !59
  %96 = fadd contract double %95, %"bias::Float64", !dbg !59
  %97 = fcmp ule double %96, 1.250000e-01, !dbg !62
  %value_phi11.p.v.1.1 = select i1 %97, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.1.1 = fmul double %96, %value_phi11.p.v.1.1, !dbg !65
  %value_phi11.1.1 = fadd double %value_phi11.145, %value_phi11.p.1.1, !dbg !65
  %98 = fmul contract double %value_phi11.1.1, 0x3FF000001FF19E24, !dbg !59
  %99 = fadd contract double %98, %"bias::Float64", !dbg !59
  %100 = fcmp ule double %99, 1.250000e-01, !dbg !62
  %value_phi11.p.v.2.1 = select i1 %100, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.2.1 = fmul double %99, %value_phi11.p.v.2.1, !dbg !65
  %value_phi11.2.1 = fadd double %value_phi11.1.1, %value_phi11.p.2.1, !dbg !65
  %101 = fmul contract double %value_phi11.2.1, 0x3FF000001FF19E24, !dbg !59
  %102 = fadd contract double %101, %"bias::Float64", !dbg !59
  %103 = fcmp ule double %102, 1.250000e-01, !dbg !62
  %value_phi11.p.v.3.1 = select i1 %103, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.3.1 = fmul double %102, %value_phi11.p.v.3.1, !dbg !65
  %value_phi11.3.1 = fadd double %value_phi11.2.1, %value_phi11.p.3.1, !dbg !65
  %104 = fmul contract double %value_phi11.3.1, 0x3FF000001FF19E24, !dbg !59
  %105 = fadd contract double %104, %"bias::Float64", !dbg !59
  %106 = fcmp ule double %105, 1.250000e-01, !dbg !62
  %value_phi11.p.v.4.1 = select i1 %106, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.4.1 = fmul double %105, %value_phi11.p.v.4.1, !dbg !65
  %value_phi11.4.1 = fadd double %value_phi11.3.1, %value_phi11.p.4.1, !dbg !65
  %107 = fmul contract double %value_phi11.4.1, 0x3FF000001FF19E24, !dbg !59
  %108 = fadd contract double %107, %"bias::Float64", !dbg !59
  %109 = fcmp ule double %108, 1.250000e-01, !dbg !62
  %value_phi11.p.v.5.1 = select i1 %109, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.5.1 = fmul double %108, %value_phi11.p.v.5.1, !dbg !65
  %value_phi11.5.1 = fadd double %value_phi11.4.1, %value_phi11.p.5.1, !dbg !65
  %110 = fmul contract double %value_phi11.5.1, 0x3FF000001FF19E24, !dbg !59
  %111 = fadd contract double %110, %"bias::Float64", !dbg !59
  %112 = fcmp ule double %111, 1.250000e-01, !dbg !62
  %value_phi11.p.v.6.1 = select i1 %112, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.6.1 = fmul double %111, %value_phi11.p.v.6.1, !dbg !65
  %value_phi11.6.1 = fadd double %value_phi11.5.1, %value_phi11.p.6.1, !dbg !65
  %113 = fmul contract double %value_phi11.6.1, 0x3FF000001FF19E24, !dbg !59
  %114 = fadd contract double %113, %"bias::Float64", !dbg !59
  %115 = fcmp ule double %114, 1.250000e-01, !dbg !62
  %value_phi11.p.v.7.1 = select i1 %115, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.7.1 = fmul double %114, %value_phi11.p.v.7.1, !dbg !65
  %value_phi11.7.1 = fadd double %value_phi11.6.1, %value_phi11.p.7.1, !dbg !65
  %niter.next.7.1 = add i64 %niter.1, 8, !dbg !40
  %niter.ncmp.7.1 = icmp eq i64 %niter.next.7.1, %unroll_iter.1, !dbg !40
  br i1 %niter.ncmp.7.1, label %L57.loopexit.unr-lcssa.1, label %L35.1, !dbg !40

L57.loopexit.unr-lcssa.1:                         ; preds = %L35.1, %L57.loopexit
  %value_phi11.lcssa.ph.1 = phi double [ undef, %L57.loopexit ], [ %value_phi11.7.1, %L35.1 ]
  %value_phi10.unr.1 = phi double [ %value_phi15, %L57.loopexit ], [ %value_phi11.7.1, %L35.1 ]
  br i1 %lcmp.mod.not, label %L57.loopexit.1, label %L35.epil.1, !dbg !40

L35.epil.1:                                       ; preds = %L57.loopexit.unr-lcssa.1, %L35.epil.1
  %value_phi10.epil.1 = phi double [ %value_phi11.epil.1, %L35.epil.1 ], [ %value_phi10.unr.1, %L57.loopexit.unr-lcssa.1 ]
  %epil.iter.1 = phi i64 [ %epil.iter.next.1, %L35.epil.1 ], [ 0, %L57.loopexit.unr-lcssa.1 ]
  %116 = fmul contract double %value_phi10.epil.1, 0x3FF000001FF19E24, !dbg !59
  %117 = fadd contract double %116, %"bias::Float64", !dbg !59
  %118 = fcmp ule double %117, 1.250000e-01, !dbg !62
  %value_phi11.p.v.epil.1 = select i1 %118, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.epil.1 = fmul double %117, %value_phi11.p.v.epil.1, !dbg !65
  %value_phi11.epil.1 = fadd double %value_phi10.epil.1, %value_phi11.p.epil.1, !dbg !65
  %epil.iter.next.1 = add i64 %epil.iter.1, 1, !dbg !40
  %epil.iter.cmp.1.not = icmp eq i64 %epil.iter.next.1, %xtraiter, !dbg !40
  br i1 %epil.iter.cmp.1.not, label %L57.loopexit.1, label %L35.epil.1, !dbg !40, !llvm.loop !66

L57.loopexit.1:                                   ; preds = %L35.epil.1, %L57.loopexit.unr-lcssa.1
  %value_phi11.lcssa.1 = phi double [ %value_phi11.lcssa.ph.1, %L57.loopexit.unr-lcssa.1 ], [ %value_phi11.epil.1, %L35.epil.1 ], !dbg !65
  %119 = fmul contract double %value_phi11.lcssa.1, 0x3FF000001FF19E24, !dbg !42
  %120 = fadd contract double %119, %"bias::Float64", !dbg !42
  %121 = fcmp ule double %120, 1.250000e-01, !dbg !48
  %value_phi15.p.v.1 = select i1 %121, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.1 = fmul double %120, %value_phi15.p.v.1, !dbg !52
  %value_phi15.1 = fadd double %value_phi11.lcssa.1, %value_phi15.p.1, !dbg !52
  %122 = call double @llvm.fabs.f64(double %120), !dbg !53
  %123 = fmul contract double %122, 1.000000e-09, !dbg !58
  %124 = fadd contract double %91, %123, !dbg !58
  br i1 %58, label %L57.loopexit.unr-lcssa.2, label %L16.new.2, !dbg !40

L16.new.2:                                        ; preds = %L57.loopexit.1
  %unroll_iter.2 = and i64 %., 9223372036854775800, !dbg !40
  br label %L35.2, !dbg !40

L35.2:                                            ; preds = %L35.2, %L16.new.2
  %value_phi10.2 = phi double [ %value_phi15.1, %L16.new.2 ], [ %value_phi11.7.2, %L35.2 ]
  %niter.2 = phi i64 [ 0, %L16.new.2 ], [ %niter.next.7.2, %L35.2 ]
  %125 = fmul contract double %value_phi10.2, 0x3FF000001FF19E24, !dbg !59
  %126 = fadd contract double %125, %"bias::Float64", !dbg !59
  %127 = fcmp ule double %126, 1.250000e-01, !dbg !62
  %value_phi11.p.v.246 = select i1 %127, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.247 = fmul double %126, %value_phi11.p.v.246, !dbg !65
  %value_phi11.248 = fadd double %value_phi10.2, %value_phi11.p.247, !dbg !65
  %128 = fmul contract double %value_phi11.248, 0x3FF000001FF19E24, !dbg !59
  %129 = fadd contract double %128, %"bias::Float64", !dbg !59
  %130 = fcmp ule double %129, 1.250000e-01, !dbg !62
  %value_phi11.p.v.1.2 = select i1 %130, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.1.2 = fmul double %129, %value_phi11.p.v.1.2, !dbg !65
  %value_phi11.1.2 = fadd double %value_phi11.248, %value_phi11.p.1.2, !dbg !65
  %131 = fmul contract double %value_phi11.1.2, 0x3FF000001FF19E24, !dbg !59
  %132 = fadd contract double %131, %"bias::Float64", !dbg !59
  %133 = fcmp ule double %132, 1.250000e-01, !dbg !62
  %value_phi11.p.v.2.2 = select i1 %133, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.2.2 = fmul double %132, %value_phi11.p.v.2.2, !dbg !65
  %value_phi11.2.2 = fadd double %value_phi11.1.2, %value_phi11.p.2.2, !dbg !65
  %134 = fmul contract double %value_phi11.2.2, 0x3FF000001FF19E24, !dbg !59
  %135 = fadd contract double %134, %"bias::Float64", !dbg !59
  %136 = fcmp ule double %135, 1.250000e-01, !dbg !62
  %value_phi11.p.v.3.2 = select i1 %136, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.3.2 = fmul double %135, %value_phi11.p.v.3.2, !dbg !65
  %value_phi11.3.2 = fadd double %value_phi11.2.2, %value_phi11.p.3.2, !dbg !65
  %137 = fmul contract double %value_phi11.3.2, 0x3FF000001FF19E24, !dbg !59
  %138 = fadd contract double %137, %"bias::Float64", !dbg !59
  %139 = fcmp ule double %138, 1.250000e-01, !dbg !62
  %value_phi11.p.v.4.2 = select i1 %139, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.4.2 = fmul double %138, %value_phi11.p.v.4.2, !dbg !65
  %value_phi11.4.2 = fadd double %value_phi11.3.2, %value_phi11.p.4.2, !dbg !65
  %140 = fmul contract double %value_phi11.4.2, 0x3FF000001FF19E24, !dbg !59
  %141 = fadd contract double %140, %"bias::Float64", !dbg !59
  %142 = fcmp ule double %141, 1.250000e-01, !dbg !62
  %value_phi11.p.v.5.2 = select i1 %142, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.5.2 = fmul double %141, %value_phi11.p.v.5.2, !dbg !65
  %value_phi11.5.2 = fadd double %value_phi11.4.2, %value_phi11.p.5.2, !dbg !65
  %143 = fmul contract double %value_phi11.5.2, 0x3FF000001FF19E24, !dbg !59
  %144 = fadd contract double %143, %"bias::Float64", !dbg !59
  %145 = fcmp ule double %144, 1.250000e-01, !dbg !62
  %value_phi11.p.v.6.2 = select i1 %145, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.6.2 = fmul double %144, %value_phi11.p.v.6.2, !dbg !65
  %value_phi11.6.2 = fadd double %value_phi11.5.2, %value_phi11.p.6.2, !dbg !65
  %146 = fmul contract double %value_phi11.6.2, 0x3FF000001FF19E24, !dbg !59
  %147 = fadd contract double %146, %"bias::Float64", !dbg !59
  %148 = fcmp ule double %147, 1.250000e-01, !dbg !62
  %value_phi11.p.v.7.2 = select i1 %148, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.7.2 = fmul double %147, %value_phi11.p.v.7.2, !dbg !65
  %value_phi11.7.2 = fadd double %value_phi11.6.2, %value_phi11.p.7.2, !dbg !65
  %niter.next.7.2 = add i64 %niter.2, 8, !dbg !40
  %niter.ncmp.7.2 = icmp eq i64 %niter.next.7.2, %unroll_iter.2, !dbg !40
  br i1 %niter.ncmp.7.2, label %L57.loopexit.unr-lcssa.2, label %L35.2, !dbg !40

L57.loopexit.unr-lcssa.2:                         ; preds = %L35.2, %L57.loopexit.1
  %value_phi11.lcssa.ph.2 = phi double [ undef, %L57.loopexit.1 ], [ %value_phi11.7.2, %L35.2 ]
  %value_phi10.unr.2 = phi double [ %value_phi15.1, %L57.loopexit.1 ], [ %value_phi11.7.2, %L35.2 ]
  br i1 %lcmp.mod.not, label %L57.loopexit.2, label %L35.epil.2, !dbg !40

L35.epil.2:                                       ; preds = %L57.loopexit.unr-lcssa.2, %L35.epil.2
  %value_phi10.epil.2 = phi double [ %value_phi11.epil.2, %L35.epil.2 ], [ %value_phi10.unr.2, %L57.loopexit.unr-lcssa.2 ]
  %epil.iter.2 = phi i64 [ %epil.iter.next.2, %L35.epil.2 ], [ 0, %L57.loopexit.unr-lcssa.2 ]
  %149 = fmul contract double %value_phi10.epil.2, 0x3FF000001FF19E24, !dbg !59
  %150 = fadd contract double %149, %"bias::Float64", !dbg !59
  %151 = fcmp ule double %150, 1.250000e-01, !dbg !62
  %value_phi11.p.v.epil.2 = select i1 %151, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.epil.2 = fmul double %150, %value_phi11.p.v.epil.2, !dbg !65
  %value_phi11.epil.2 = fadd double %value_phi10.epil.2, %value_phi11.p.epil.2, !dbg !65
  %epil.iter.next.2 = add i64 %epil.iter.2, 1, !dbg !40
  %epil.iter.cmp.2.not = icmp eq i64 %epil.iter.next.2, %xtraiter, !dbg !40
  br i1 %epil.iter.cmp.2.not, label %L57.loopexit.2, label %L35.epil.2, !dbg !40, !llvm.loop !66

L57.loopexit.2:                                   ; preds = %L35.epil.2, %L57.loopexit.unr-lcssa.2
  %value_phi11.lcssa.2 = phi double [ %value_phi11.lcssa.ph.2, %L57.loopexit.unr-lcssa.2 ], [ %value_phi11.epil.2, %L35.epil.2 ], !dbg !65
  %152 = fmul contract double %value_phi11.lcssa.2, 0x3FF000001FF19E24, !dbg !42
  %153 = fadd contract double %152, %"bias::Float64", !dbg !42
  %154 = fcmp ule double %153, 1.250000e-01, !dbg !48
  %value_phi15.p.v.2 = select i1 %154, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.2 = fmul double %153, %value_phi15.p.v.2, !dbg !52
  %value_phi15.2 = fadd double %value_phi11.lcssa.2, %value_phi15.p.2, !dbg !52
  %155 = call double @llvm.fabs.f64(double %153), !dbg !53
  %156 = fmul contract double %155, 1.000000e-09, !dbg !58
  %157 = fadd contract double %124, %156, !dbg !58
  br i1 %58, label %L57.loopexit.unr-lcssa.3, label %L16.new.3, !dbg !40

L16.new.3:                                        ; preds = %L57.loopexit.2
  %unroll_iter.3 = and i64 %., 9223372036854775800, !dbg !40
  br label %L35.3, !dbg !40

L35.3:                                            ; preds = %L35.3, %L16.new.3
  %value_phi10.3 = phi double [ %value_phi15.2, %L16.new.3 ], [ %value_phi11.7.3, %L35.3 ]
  %niter.3 = phi i64 [ 0, %L16.new.3 ], [ %niter.next.7.3, %L35.3 ]
  %158 = fmul contract double %value_phi10.3, 0x3FF000001FF19E24, !dbg !59
  %159 = fadd contract double %158, %"bias::Float64", !dbg !59
  %160 = fcmp ule double %159, 1.250000e-01, !dbg !62
  %value_phi11.p.v.349 = select i1 %160, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.350 = fmul double %159, %value_phi11.p.v.349, !dbg !65
  %value_phi11.351 = fadd double %value_phi10.3, %value_phi11.p.350, !dbg !65
  %161 = fmul contract double %value_phi11.351, 0x3FF000001FF19E24, !dbg !59
  %162 = fadd contract double %161, %"bias::Float64", !dbg !59
  %163 = fcmp ule double %162, 1.250000e-01, !dbg !62
  %value_phi11.p.v.1.3 = select i1 %163, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.1.3 = fmul double %162, %value_phi11.p.v.1.3, !dbg !65
  %value_phi11.1.3 = fadd double %value_phi11.351, %value_phi11.p.1.3, !dbg !65
  %164 = fmul contract double %value_phi11.1.3, 0x3FF000001FF19E24, !dbg !59
  %165 = fadd contract double %164, %"bias::Float64", !dbg !59
  %166 = fcmp ule double %165, 1.250000e-01, !dbg !62
  %value_phi11.p.v.2.3 = select i1 %166, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.2.3 = fmul double %165, %value_phi11.p.v.2.3, !dbg !65
  %value_phi11.2.3 = fadd double %value_phi11.1.3, %value_phi11.p.2.3, !dbg !65
  %167 = fmul contract double %value_phi11.2.3, 0x3FF000001FF19E24, !dbg !59
  %168 = fadd contract double %167, %"bias::Float64", !dbg !59
  %169 = fcmp ule double %168, 1.250000e-01, !dbg !62
  %value_phi11.p.v.3.3 = select i1 %169, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.3.3 = fmul double %168, %value_phi11.p.v.3.3, !dbg !65
  %value_phi11.3.3 = fadd double %value_phi11.2.3, %value_phi11.p.3.3, !dbg !65
  %170 = fmul contract double %value_phi11.3.3, 0x3FF000001FF19E24, !dbg !59
  %171 = fadd contract double %170, %"bias::Float64", !dbg !59
  %172 = fcmp ule double %171, 1.250000e-01, !dbg !62
  %value_phi11.p.v.4.3 = select i1 %172, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.4.3 = fmul double %171, %value_phi11.p.v.4.3, !dbg !65
  %value_phi11.4.3 = fadd double %value_phi11.3.3, %value_phi11.p.4.3, !dbg !65
  %173 = fmul contract double %value_phi11.4.3, 0x3FF000001FF19E24, !dbg !59
  %174 = fadd contract double %173, %"bias::Float64", !dbg !59
  %175 = fcmp ule double %174, 1.250000e-01, !dbg !62
  %value_phi11.p.v.5.3 = select i1 %175, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.5.3 = fmul double %174, %value_phi11.p.v.5.3, !dbg !65
  %value_phi11.5.3 = fadd double %value_phi11.4.3, %value_phi11.p.5.3, !dbg !65
  %176 = fmul contract double %value_phi11.5.3, 0x3FF000001FF19E24, !dbg !59
  %177 = fadd contract double %176, %"bias::Float64", !dbg !59
  %178 = fcmp ule double %177, 1.250000e-01, !dbg !62
  %value_phi11.p.v.6.3 = select i1 %178, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.6.3 = fmul double %177, %value_phi11.p.v.6.3, !dbg !65
  %value_phi11.6.3 = fadd double %value_phi11.5.3, %value_phi11.p.6.3, !dbg !65
  %179 = fmul contract double %value_phi11.6.3, 0x3FF000001FF19E24, !dbg !59
  %180 = fadd contract double %179, %"bias::Float64", !dbg !59
  %181 = fcmp ule double %180, 1.250000e-01, !dbg !62
  %value_phi11.p.v.7.3 = select i1 %181, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.7.3 = fmul double %180, %value_phi11.p.v.7.3, !dbg !65
  %value_phi11.7.3 = fadd double %value_phi11.6.3, %value_phi11.p.7.3, !dbg !65
  %niter.next.7.3 = add i64 %niter.3, 8, !dbg !40
  %niter.ncmp.7.3 = icmp eq i64 %niter.next.7.3, %unroll_iter.3, !dbg !40
  br i1 %niter.ncmp.7.3, label %L57.loopexit.unr-lcssa.3, label %L35.3, !dbg !40

L57.loopexit.unr-lcssa.3:                         ; preds = %L35.3, %L57.loopexit.2
  %value_phi11.lcssa.ph.3 = phi double [ undef, %L57.loopexit.2 ], [ %value_phi11.7.3, %L35.3 ]
  %value_phi10.unr.3 = phi double [ %value_phi15.2, %L57.loopexit.2 ], [ %value_phi11.7.3, %L35.3 ]
  br i1 %lcmp.mod.not, label %L57.loopexit.3, label %L35.epil.3, !dbg !40

L35.epil.3:                                       ; preds = %L57.loopexit.unr-lcssa.3, %L35.epil.3
  %value_phi10.epil.3 = phi double [ %value_phi11.epil.3, %L35.epil.3 ], [ %value_phi10.unr.3, %L57.loopexit.unr-lcssa.3 ]
  %epil.iter.3 = phi i64 [ %epil.iter.next.3, %L35.epil.3 ], [ 0, %L57.loopexit.unr-lcssa.3 ]
  %182 = fmul contract double %value_phi10.epil.3, 0x3FF000001FF19E24, !dbg !59
  %183 = fadd contract double %182, %"bias::Float64", !dbg !59
  %184 = fcmp ule double %183, 1.250000e-01, !dbg !62
  %value_phi11.p.v.epil.3 = select i1 %184, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.epil.3 = fmul double %183, %value_phi11.p.v.epil.3, !dbg !65
  %value_phi11.epil.3 = fadd double %value_phi10.epil.3, %value_phi11.p.epil.3, !dbg !65
  %epil.iter.next.3 = add i64 %epil.iter.3, 1, !dbg !40
  %epil.iter.cmp.3.not = icmp eq i64 %epil.iter.next.3, %xtraiter, !dbg !40
  br i1 %epil.iter.cmp.3.not, label %L57.loopexit.3, label %L35.epil.3, !dbg !40, !llvm.loop !66

L57.loopexit.3:                                   ; preds = %L35.epil.3, %L57.loopexit.unr-lcssa.3
  %value_phi11.lcssa.3 = phi double [ %value_phi11.lcssa.ph.3, %L57.loopexit.unr-lcssa.3 ], [ %value_phi11.epil.3, %L35.epil.3 ], !dbg !65
  %185 = fmul contract double %value_phi11.lcssa.3, 0x3FF000001FF19E24, !dbg !42
  %186 = fadd contract double %185, %"bias::Float64", !dbg !42
  %187 = fcmp ule double %186, 1.250000e-01, !dbg !48
  %value_phi15.p.v.3 = select i1 %187, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.3 = fmul double %186, %value_phi15.p.v.3, !dbg !52
  %value_phi15.3 = fadd double %value_phi11.lcssa.3, %value_phi15.p.3, !dbg !52
  %188 = call double @llvm.fabs.f64(double %186), !dbg !53
  %189 = fmul contract double %188, 1.000000e-09, !dbg !58
  %190 = fadd contract double %157, %189, !dbg !58
  %niter42.next.3 = add i64 %niter42, 4, !dbg !41
  %niter42.ncmp.3 = icmp eq i64 %niter42.next.3, %unroll_iter41, !dbg !41
  br i1 %niter42.ncmp.3, label %L80.loopexit27.unr-lcssa, label %L16, !dbg !41

L80.loopexit.unr-lcssa:                           ; preds = %L16.us, %L16.us.preheader
  %value_phi15.us.lcssa.ph = phi double [ undef, %L16.us.preheader ], [ %value_phi15.us.7, %L16.us ]
  %.lcssa.ph = phi double [ undef, %L16.us.preheader ], [ %57, %L16.us ]
  %value_phi4.us.unr = phi double [ 0.000000e+00, %L16.us.preheader ], [ %57, %L16.us ]
  %value_phi5.us.unr = phi double [ %"x::Float64", %L16.us.preheader ], [ %value_phi15.us.7, %L16.us ]
  %lcmp.mod54.not = icmp eq i64 %xtraiter52, 0, !dbg !41
  br i1 %lcmp.mod54.not, label %L80, label %L16.us.epil, !dbg !41

L16.us.epil:                                      ; preds = %L80.loopexit.unr-lcssa, %L16.us.epil
  %value_phi4.us.epil = phi double [ %196, %L16.us.epil ], [ %value_phi4.us.unr, %L80.loopexit.unr-lcssa ]
  %value_phi5.us.epil = phi double [ %value_phi15.us.epil, %L16.us.epil ], [ %value_phi5.us.unr, %L80.loopexit.unr-lcssa ]
  %epil.iter53 = phi i64 [ %epil.iter53.next, %L16.us.epil ], [ 0, %L80.loopexit.unr-lcssa ]
  %191 = fmul contract double %value_phi5.us.epil, 0x3FF000001FF19E24, !dbg !42
  %192 = fadd contract double %191, %"bias::Float64", !dbg !42
  %193 = fcmp ule double %192, 1.250000e-01, !dbg !48
  %value_phi15.p.v.us.epil = select i1 %193, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.us.epil = fmul double %192, %value_phi15.p.v.us.epil, !dbg !52
  %value_phi15.us.epil = fadd double %value_phi5.us.epil, %value_phi15.p.us.epil, !dbg !52
  %194 = call double @llvm.fabs.f64(double %192), !dbg !53
  %195 = fmul contract double %194, 1.000000e-09, !dbg !58
  %196 = fadd contract double %value_phi4.us.epil, %195, !dbg !58
  %epil.iter53.next = add i64 %epil.iter53, 1, !dbg !41
  %epil.iter53.cmp.not = icmp eq i64 %epil.iter53.next, %xtraiter52, !dbg !41
  br i1 %epil.iter53.cmp.not, label %L80, label %L16.us.epil, !dbg !41, !llvm.loop !68

L80.loopexit27.unr-lcssa:                         ; preds = %L57.loopexit.3, %L16.preheader26
  %value_phi15.lcssa.ph = phi double [ undef, %L16.preheader26 ], [ %value_phi15.3, %L57.loopexit.3 ]
  %.lcssa28.ph = phi double [ undef, %L16.preheader26 ], [ %190, %L57.loopexit.3 ]
  %value_phi4.unr = phi double [ 0.000000e+00, %L16.preheader26 ], [ %190, %L57.loopexit.3 ]
  %value_phi5.unr = phi double [ %"x::Float64", %L16.preheader26 ], [ %value_phi15.3, %L57.loopexit.3 ]
  %lcmp.mod38.not = icmp eq i64 %xtraiter30, 0, !dbg !40
  br i1 %lcmp.mod38.not, label %L80, label %L16.epil, !dbg !40

L16.epil:                                         ; preds = %L80.loopexit27.unr-lcssa, %L57.loopexit.epil
  %value_phi4.epil = phi double [ %230, %L57.loopexit.epil ], [ %value_phi4.unr, %L80.loopexit27.unr-lcssa ]
  %value_phi5.epil = phi double [ %value_phi15.epil, %L57.loopexit.epil ], [ %value_phi5.unr, %L80.loopexit27.unr-lcssa ]
  %epil.iter37 = phi i64 [ %epil.iter37.next, %L57.loopexit.epil ], [ 0, %L80.loopexit27.unr-lcssa ]
  %xtraiter.epil = and i64 %., 7, !dbg !40
  %197 = icmp ult i64 %5, 7, !dbg !40
  br i1 %197, label %L57.loopexit.unr-lcssa.epil, label %L16.new.epil, !dbg !40

L16.new.epil:                                     ; preds = %L16.epil
  %unroll_iter.epil = and i64 %., 9223372036854775800, !dbg !40
  br label %L35.epil31, !dbg !40

L35.epil31:                                       ; preds = %L35.epil31, %L16.new.epil
  %value_phi10.epil33 = phi double [ %value_phi5.epil, %L16.new.epil ], [ %value_phi11.7.epil, %L35.epil31 ]
  %niter.epil = phi i64 [ 0, %L16.new.epil ], [ %niter.next.7.epil, %L35.epil31 ]
  %198 = fmul contract double %value_phi10.epil33, 0x3FF000001FF19E24, !dbg !59
  %199 = fadd contract double %198, %"bias::Float64", !dbg !59
  %200 = fcmp ule double %199, 1.250000e-01, !dbg !62
  %value_phi11.p.v.epil34 = select i1 %200, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.epil35 = fmul double %199, %value_phi11.p.v.epil34, !dbg !65
  %value_phi11.epil36 = fadd double %value_phi10.epil33, %value_phi11.p.epil35, !dbg !65
  %201 = fmul contract double %value_phi11.epil36, 0x3FF000001FF19E24, !dbg !59
  %202 = fadd contract double %201, %"bias::Float64", !dbg !59
  %203 = fcmp ule double %202, 1.250000e-01, !dbg !62
  %value_phi11.p.v.1.epil = select i1 %203, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.1.epil = fmul double %202, %value_phi11.p.v.1.epil, !dbg !65
  %value_phi11.1.epil = fadd double %value_phi11.epil36, %value_phi11.p.1.epil, !dbg !65
  %204 = fmul contract double %value_phi11.1.epil, 0x3FF000001FF19E24, !dbg !59
  %205 = fadd contract double %204, %"bias::Float64", !dbg !59
  %206 = fcmp ule double %205, 1.250000e-01, !dbg !62
  %value_phi11.p.v.2.epil = select i1 %206, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.2.epil = fmul double %205, %value_phi11.p.v.2.epil, !dbg !65
  %value_phi11.2.epil = fadd double %value_phi11.1.epil, %value_phi11.p.2.epil, !dbg !65
  %207 = fmul contract double %value_phi11.2.epil, 0x3FF000001FF19E24, !dbg !59
  %208 = fadd contract double %207, %"bias::Float64", !dbg !59
  %209 = fcmp ule double %208, 1.250000e-01, !dbg !62
  %value_phi11.p.v.3.epil = select i1 %209, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.3.epil = fmul double %208, %value_phi11.p.v.3.epil, !dbg !65
  %value_phi11.3.epil = fadd double %value_phi11.2.epil, %value_phi11.p.3.epil, !dbg !65
  %210 = fmul contract double %value_phi11.3.epil, 0x3FF000001FF19E24, !dbg !59
  %211 = fadd contract double %210, %"bias::Float64", !dbg !59
  %212 = fcmp ule double %211, 1.250000e-01, !dbg !62
  %value_phi11.p.v.4.epil = select i1 %212, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.4.epil = fmul double %211, %value_phi11.p.v.4.epil, !dbg !65
  %value_phi11.4.epil = fadd double %value_phi11.3.epil, %value_phi11.p.4.epil, !dbg !65
  %213 = fmul contract double %value_phi11.4.epil, 0x3FF000001FF19E24, !dbg !59
  %214 = fadd contract double %213, %"bias::Float64", !dbg !59
  %215 = fcmp ule double %214, 1.250000e-01, !dbg !62
  %value_phi11.p.v.5.epil = select i1 %215, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.5.epil = fmul double %214, %value_phi11.p.v.5.epil, !dbg !65
  %value_phi11.5.epil = fadd double %value_phi11.4.epil, %value_phi11.p.5.epil, !dbg !65
  %216 = fmul contract double %value_phi11.5.epil, 0x3FF000001FF19E24, !dbg !59
  %217 = fadd contract double %216, %"bias::Float64", !dbg !59
  %218 = fcmp ule double %217, 1.250000e-01, !dbg !62
  %value_phi11.p.v.6.epil = select i1 %218, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.6.epil = fmul double %217, %value_phi11.p.v.6.epil, !dbg !65
  %value_phi11.6.epil = fadd double %value_phi11.5.epil, %value_phi11.p.6.epil, !dbg !65
  %219 = fmul contract double %value_phi11.6.epil, 0x3FF000001FF19E24, !dbg !59
  %220 = fadd contract double %219, %"bias::Float64", !dbg !59
  %221 = fcmp ule double %220, 1.250000e-01, !dbg !62
  %value_phi11.p.v.7.epil = select i1 %221, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.7.epil = fmul double %220, %value_phi11.p.v.7.epil, !dbg !65
  %value_phi11.7.epil = fadd double %value_phi11.6.epil, %value_phi11.p.7.epil, !dbg !65
  %niter.next.7.epil = add i64 %niter.epil, 8, !dbg !40
  %niter.ncmp.7.epil = icmp eq i64 %niter.next.7.epil, %unroll_iter.epil, !dbg !40
  br i1 %niter.ncmp.7.epil, label %L57.loopexit.unr-lcssa.epil, label %L35.epil31, !dbg !40

L57.loopexit.unr-lcssa.epil:                      ; preds = %L35.epil31, %L16.epil
  %value_phi11.lcssa.ph.epil = phi double [ undef, %L16.epil ], [ %value_phi11.7.epil, %L35.epil31 ]
  %value_phi10.unr.epil = phi double [ %value_phi5.epil, %L16.epil ], [ %value_phi11.7.epil, %L35.epil31 ]
  %lcmp.mod.epil.not = icmp eq i64 %xtraiter.epil, 0, !dbg !40
  br i1 %lcmp.mod.epil.not, label %L57.loopexit.epil, label %L35.epil.epil, !dbg !40

L35.epil.epil:                                    ; preds = %L57.loopexit.unr-lcssa.epil, %L35.epil.epil
  %value_phi10.epil.epil = phi double [ %value_phi11.epil.epil, %L35.epil.epil ], [ %value_phi10.unr.epil, %L57.loopexit.unr-lcssa.epil ]
  %epil.iter.epil = phi i64 [ %epil.iter.next.epil, %L35.epil.epil ], [ 0, %L57.loopexit.unr-lcssa.epil ]
  %222 = fmul contract double %value_phi10.epil.epil, 0x3FF000001FF19E24, !dbg !59
  %223 = fadd contract double %222, %"bias::Float64", !dbg !59
  %224 = fcmp ule double %223, 1.250000e-01, !dbg !62
  %value_phi11.p.v.epil.epil = select i1 %224, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !65
  %value_phi11.p.epil.epil = fmul double %223, %value_phi11.p.v.epil.epil, !dbg !65
  %value_phi11.epil.epil = fadd double %value_phi10.epil.epil, %value_phi11.p.epil.epil, !dbg !65
  %epil.iter.next.epil = add i64 %epil.iter.epil, 1, !dbg !40
  %epil.iter.cmp.epil.not = icmp eq i64 %epil.iter.next.epil, %xtraiter.epil, !dbg !40
  br i1 %epil.iter.cmp.epil.not, label %L57.loopexit.epil, label %L35.epil.epil, !dbg !40, !llvm.loop !66

L57.loopexit.epil:                                ; preds = %L35.epil.epil, %L57.loopexit.unr-lcssa.epil
  %value_phi11.lcssa.epil = phi double [ %value_phi11.lcssa.ph.epil, %L57.loopexit.unr-lcssa.epil ], [ %value_phi11.epil.epil, %L35.epil.epil ], !dbg !65
  %225 = fmul contract double %value_phi11.lcssa.epil, 0x3FF000001FF19E24, !dbg !42
  %226 = fadd contract double %225, %"bias::Float64", !dbg !42
  %227 = fcmp ule double %226, 1.250000e-01, !dbg !48
  %value_phi15.p.v.epil = select i1 %227, double -5.000000e-07, double 0x3EB0C6F7A0B5ED8D, !dbg !52
  %value_phi15.p.epil = fmul double %226, %value_phi15.p.v.epil, !dbg !52
  %value_phi15.epil = fadd double %value_phi11.lcssa.epil, %value_phi15.p.epil, !dbg !52
  %228 = call double @llvm.fabs.f64(double %226), !dbg !53
  %229 = fmul contract double %228, 1.000000e-09, !dbg !58
  %230 = fadd contract double %value_phi4.epil, %229, !dbg !58
  %epil.iter37.next = add i64 %epil.iter37, 1, !dbg !41
  %epil.iter37.cmp.not = icmp eq i64 %epil.iter37.next, %xtraiter30, !dbg !41
  br i1 %epil.iter37.cmp.not, label %L80, label %L16.epil, !dbg !41, !llvm.loop !69

L80:                                              ; preds = %L80.loopexit27.unr-lcssa, %L57.loopexit.epil, %L80.loopexit.unr-lcssa, %L16.us.epil, %top
  %value_phi18 = phi double [ 0.000000e+00, %top ], [ %.lcssa.ph, %L80.loopexit.unr-lcssa ], [ %196, %L16.us.epil ], [ %.lcssa28.ph, %L80.loopexit27.unr-lcssa ], [ %230, %L57.loopexit.epil ]
  %value_phi19 = phi double [ %"x::Float64", %top ], [ %value_phi15.us.lcssa.ph, %L80.loopexit.unr-lcssa ], [ %value_phi15.us.epil, %L16.us.epil ], [ %value_phi15.lcssa.ph, %L80.loopexit27.unr-lcssa ], [ %value_phi15.epil, %L57.loopexit.epil ]
  %231 = fadd double %value_phi18, %value_phi19, !dbg !70
  ret double %231, !dbg !72
}
