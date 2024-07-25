	.text
	.file	"LLVMDialectModule"
	.globl	add                             # -- Begin function add
	.p2align	4, 0x90
	.type	add,@function
add:                                    # @add
# %bb.0:
	leaq	(%rdi,%rsi), %rax
	retq
.Lfunc_end0:
	.size	add, .Lfunc_end0-add
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
# %bb.0:
	movl	$5, %eax
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
	.addrsig
