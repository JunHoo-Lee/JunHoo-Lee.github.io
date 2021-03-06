# Head.S init

start from __HEAD

```c
#define __HEAD		.section	".head.text","ax"
// at init.h
// eventually it reads head.text at linker script
컴파일을 어디다 할지 지정해주는 것, 위치를 __head.text로 모아줘!
```

preserve_boot_args

```assembly
SYM_CODE_START_LOCAL(preserve_boot_args)
// by bootloader, FBT is saved in x0
	mov	x21, x0    // x21=FDT
    // adr_l : calculate the location of symbol with PC
    // since ADRP gets by 4K(Page) we adr_l adds lower 12 bit
	adr_l	x0, boot_args			// record the contents of
    // STP X9, X8 [x4]: Store the doubleword in X9 to address X4 and
	// stores the doubleword in X8 to address X4 + 8
	stp	x21, x1, [x0]			// x0 .. x3 at kernel entry
	stp	x2, x3, [x0, #16]
    // memory barrrier
	dmb	sy				// needed before dc ivac with
	mov	x1, #0x20			// 4 x 8 bytes

	mov	x1, #0x20			// 4 x 8 bytes
	b	__inval_dcache_area		// tail call
SYM_CODE_END(preserve_boot_args)

```

## inval_dcache_area 
```c
SYM_FUNC_START_PI(__inval_dcache_area)
	/* FALLTHROUGH */

// [iamroot]
// $kernel/arch/arm64/include/asm/linkage.h

// 1. SYM_FUNC_START_PI:
// #define SYM_FUNC_START_PI(x)			\
//		SYM_FUNC_START_ALIAS(__pi_##x);	\
//		SYM_FUNC_START(x)
// 2.SYM_FUNC_START_LOCAL:
// #define SYM_FUNC_START_LOCAL(name)			\
//	SYM_START(name, SYM_L_LOCAL, SYM_A_ALIGN)	\
//  BTI_C




/*
 *	__dma_inv_area(start, size)
 *	- start   - virtual start address of region
 *	- size    - size in question
 */
	// x1 <- boot_args의 크기 (#32)
	// x0 <- (boot_args[0])

	add	x1, x1, x0			// x1 <- boot_args 의 끝점

	// assembler.h:294
	dcache_line_size x2, x3
	sub	x3, x2, #1
	tst	x1, x3				// end cache line aligned?
	bic	x1, x1, x3
	b.eq	1f
	dc	civac, x1			// clean & invalidate D / U line
1:	tst	x0, x3				// start cache line aligned?
	bic	x0, x0, x3
	b.eq	2f
	dc	civac, x0			// clean & invalidate D / U line
	b	3f
2:	dc	ivac, x0			// invalidate D / U line
3:	add	x0, x0, x2
	cmp	x0, x1
	b.lo	2b
	dsb	sy
	ret
SYM_FUNC_END_PI(__inval_dcache_area)
```

## dcache_line_size
```c

/*
 * dcache_line_size - get the safe D-cache line size across all CPUs
 */
	.macro	dcache_line_size, reg, tmp

	// read_ctr: line 265

	read_ctr	\tmp
	ubfm		\tmp, \tmp, #16, #19	// cache line size encoding
	mov		\reg, #4		// bytes per word
	lsl		\reg, \reg, \tmp	// actual cache line size
	.endm

```
C 6.2.333
ubfm = unsigned bit field move
[ubfm 설명](https://developer.arm.com/documentation/dui0802/a/A64-General-Instructions/UBFM)

lsl = logical left shift

## read_ctr

```c

/*
 * read_ctr - read CTR_EL0. If the system has mismatched register fields,
 * provide the system wide safe value from arm64_ftr_reg_ctrel0.sys_val
 */
	.macro	read_ctr, reg
	// CTR_EL0: D13.2.33 Cache Type Reigster
	// Provides information about the architecture of the caches.
	// alternative_if_not: arch/arm64/include/asm/alternative.h:150

alternative_if_not ARM64_MISMATCHED_CACHE_TYPE
	mrs	\reg, ctr_el0			// read CTR
	nop
alternative_else
	ldr_l	\reg, arm64_ftr_reg_ctrel0 + ARM64_FTR_SYSVAL
alternative_endif
	.endm
```

## with . -> it is GNU_RELATED code, the other is arm assembly

## alternative_if_not_cap


```c
.macro alternative_if_not cap
// .Lasm_alt_mode: Local label(asm_alt_mode)
	// .set: set the value of symbol to expression.
	//       This changes symbol’s value and type to conform to expression.
	// .pushsection
	// .popsection

	// .altinstructions:
	// "kernel/vmlinux.lds.S" 에 altinstructions 정의 되어 있어요
	// .altinstructions : { 	__alt_instructions = .;
	// *(.altinstructions) __alt_instructions_end = .; }

	// local label asm_alt_mode를 0으로 설정
	// .set adams, 0x2A -> adams = 0b00101010
	// asm_alt_mode -> 0
	.set .Lasm_alt_mode, 0

	// .pushsection name ("flag") (@type) (arguments)
	//
	// altinstruction  섹션 추가하고 플래그를 "a"로 설정
	// kernel/vmlinux.lds.S에 정의되어 있는 부분을
	// "a(SHF_ALLOC: the section is allocatable.)"

	.pushsection .altinstructions, "a"

	altinstruction_entry 661f, 663f, \cap, 662f-661f, 664f-663f
	.popsection
```
[pushsection 사용하는 이유](http://rsusu1.rnd.runnet.ru/linux/doc/ref/sec_stck.htm)

[flag 참고 자료](https://www.keil.com/support/man/docs/armclang_ref/armclang_ref_bpl1510589893923.htm)

alt instruction

Roughly, a section is a range of addresses, with no gaps; all data “in” those addresses is treated the same for some particular purpose. For example there may be a “read only” section.

[section 설명](https://sourceware.org/binutils/docs/as/Secs-Background.html#Secs-Background)

[.word 설명](https://sourceware.org/binutils/docs/as/Word.html#Word)

```c
.macro altinstruction_entry orig_offset alt_offset feature orig_len alt_len
	// ,word 0xDEADBEEF: 워드(4 bytes) 추가
	.word \orig_offset - .
	.word \alt_offset - .
	.hword \feature
	.byte \orig_len
	.byte \alt_len
.endm

```

Questions:
1. [PTE-review](#pte-review)
2. [Identity-mapping](#identity-mapping)
3. Show 