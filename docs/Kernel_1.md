## Make page Table

func __create_page_tables

It makes page table for activate MMV

1. invalidate the init page table
```
    adrp x0, init_pg_dir
    #copy init_pg_dir adr to x0
    adrp	x1, init_pg_end
    sub	x1, x1, x0
    #x0 is init_pg_dir adr, x0 is total code size of pg_dir
    bl	__inval_dcache_area
    # clear dcache
```
## Why adrp?
- relative, compare to PC, compile saves as offset
- We are now PA, using page table, it starts at FF, it cannot be run..... So we should you relative PC.

adrp operation

```
Operation
bits(64) base = PC[];
base<11:0> = Zeros(12);
X[d] = base + imm;
```

```
    /*
    * Clear the init page tables.
    */
    adrp	x0, init_pg_dir
    adrp	x1, init_pg_end
    sub	x1, x1, x0
    #stp : store to pair of registers
1:	stp	xzr, xzr, [x0], #16
    stp	xzr, xzr, [x0], #16
    stp	xzr, xzr, [x0], #16
    stp	xzr, xzr, [x0], #16
    clear PTE to 0
```
## PTE review
```
    subs	x1, x1, #64
    b.ne	1b
```

```
	mov	x7, SWAPPER_MM_MMUFLAGS

	/*
	 * Create the identity mapping.
	 */
	adrp	x0, idmap_pg_dir
	adrp	x3, __idmap_text_start		// __pa(__idmap_text_start)

```
## Identity Mapping

```assembly
#ifdef CONFIG_ARM64_VA_BITS_52 // page size => 4k
	mrs_s	x6, SYS_ID_AA64MMFR2_EL1
    //  MRS R1, SCTLR
    //  writes the contents of the CP15  coprocessor
    //  register SCTLR into R1
	and	x6, x6, #(0xf << ID_AA64MMFR2_LVA_SHIFT)
	mov	x5, #52
	cbnz	x6, 1f
    // CBNZ Compare and branch if nonzero
    // jump to next 1
#endif
	mov	x5, #VA_BITS_MIN
    // if under 48
1:
	adr_l	x6, vabits_actual // c-symbol
    // adr_l, -> it is read symbol! macro...
	str	x5, [x6]
	dmb	sy
    // Clean and Invalidate by Virtual Address to Point of Coherency(consistency)
	dc	ivac, x6		// Invalidate potentially stale cache line
```













Questions:
1. [PTE-review](#pte-review)
2. [Identity-mapping](#identity-mapping)
3. Show 