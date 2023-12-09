/*
 * CSE 351 Lab 5 (Dynamic Storage Allocator)
 *
 * Name(s):  Luke Sorvik 
 * NetID(s): lukesorv
 *
 * NOTES:
 *  - Explicit allocator with an explicit free-list
 *  - Free-list uses a single, doubly-linked list with LIFO insertion policy,
 *    first-fit search strategy, and immediate coalescing.
 *  - We use "next" and "previous" to refer to blocks as ordered in the free-list.
 *  - We use "following" and "preceding" to refer to adjacent blocks in memory.
 *  - Pointers in the free-list will point to the beginning of a heap block
 *    (i.e., to the header).
 *  - Pointers returned by mm_malloc point to the beginning of the payload
 *    (i.e., to the word after the header).
 *
 * ALLOCATOR BLOCKS:
 *  - See definition of block_info struct fields further down
 *  - USED: +---------------+   FREE: +---------------+
 *          |    header     |         |    header     |
 *          |(size_and_tags)|         |(size_and_tags)|
 *          +---------------+         +---------------+
 *          |  payload and  |         |   next ptr    |
 *          |    padding    |         +---------------+
 *          |       .       |         |   prev ptr    |
 *          |       .       |         +---------------+
 *          |       .       |         |  free space   |
 *          |               |         |  and padding  |
 *          |               |         |      ...      |
 *          |               |         +---------------+
 *          |               |         |    footer     |
 *          |               |         |(size_and_tags)|
 *          +---------------+         +---------------+
 *
 * BOUNDARY TAGS:
 *  - Headers and footers for a heap block store identical information.
 *  - The block size is stored as a word, but because of alignment, we can use
 *    some number of the least significant bits as tags/flags.
 *  - TAG_USED is bit 0 (the 1's digit) and indicates if this heap block is
 *    used/allocated.
 *  - TAG_PRECEDING_USED is bit 1 (the 2's digit) and indicates if the
 *    preceding heap block is used/allocated. Used for coalescing and avoids
 *    the need for a footer in used/allocated blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"


// Static functions for unscaled pointer arithmetic to keep other code cleaner.
//  - The first argument is void* to enable you to pass in any type of pointer
//  - Casting to char* changes the pointer arithmetic scaling to 1 byte
//    (e.g., UNSCALED_POINTER_ADD(0x1, 1) returns 0x2)
//  - We cast the result to void* to force you to cast back to the appropriate
//    type and ensure you don't accidentally use the resulting pointer as a
//    char* implicitly.
static inline void* UNSCALED_POINTER_ADD(void* p, int x) { return ((void*)((char*)(p) + (x))); }
static inline void* UNSCALED_POINTER_SUB(void* p, int x) { return ((void*)((char*)(p) - (x))); }


// A block_info can be used to access information about a heap block,
// including boundary tag info (size and usage tags in header and footer)
// and pointers to the next and previous blocks in the free-list.
struct block_info {
    // Size of the block and tags (preceding-used? and used? flags) combined
	// together. See the SIZE() function and TAG macros below for more details
	// and how to extract these pieces of info.
    size_t size_and_tags;
    // Pointer to the next block in the free list.
    struct block_info* next;
    // Pointer to the previous block in the free list.
    struct block_info* prev;
};
typedef struct block_info block_info;


// Pointer to the first block_info in the free list, the list's head.
// In this implementation, this is stored in the first word in the heap and
// accessed via mem_heap_lo().
#define FREE_LIST_HEAD *((block_info **)mem_heap_lo())

// Size of a word on this architecture.
#define WORD_SIZE sizeof(void*)

// Minimum block size (accounts for header, next ptr, prev ptr, and footer).
#define MIN_BLOCK_SIZE (sizeof(block_info) + WORD_SIZE)

// Alignment requirement for allocator.
#define ALIGNMENT 8

// SIZE(block_info->size_and_tags) extracts the size of a 'size_and_tags' field.
// SIZE(size) returns a properly-aligned value of 'size' (by rounding down).
static inline size_t SIZE(size_t x) { return ((x) & ~(ALIGNMENT - 1)); }

// Bit mask to use to extract or set TAG_USED in a boundary tag.
#define TAG_USED 1

// Bit mask to use to extract or set TAG_PRECEDING_USED in a boundary tag.
#define TAG_PRECEDING_USED 2


/*
 * Print the heap by iterating through it as an implicit free list.
 *  - For debugging; make sure to remove calls before submission as will affect
 *    throughput.
 *  - Can ignore compiler warning about this function being unused.
 */
static void examine_heap() {
  block_info* block;

  // print to stderr so output isn't buffered and not output if we crash
  fprintf(stderr, "FREE_LIST_HEAD: %p\n", (void*) FREE_LIST_HEAD);

  for (block = (block_info*) UNSCALED_POINTER_ADD(mem_heap_lo(), WORD_SIZE);  // first block on heap
       SIZE(block->size_and_tags) != 0 && block < (block_info*) mem_heap_hi();
       block = (block_info*) UNSCALED_POINTER_ADD(block, SIZE(block->size_and_tags))) {

    // print out common block attributes
    fprintf(stderr, "%p: %ld %ld %ld\t",
            (void*) block,
            SIZE(block->size_and_tags),
            block->size_and_tags & TAG_PRECEDING_USED,
            block->size_and_tags & TAG_USED);

    // and allocated/free specific data
    if (block->size_and_tags & TAG_USED) {
      fprintf(stderr, "ALLOCATED\n");
    } else {
      fprintf(stderr, "FREE\tnext: %p, prev: %p\n",
              (void*) block->next,
              (void*) block->prev);
    }
  }
  fprintf(stderr, "END OF HEAP\n\n");
}


/*
 * Find a free block of the requested size in the free list.
 * Returns NULL if no free block is large enough.
 */
static block_info* search_free_list(size_t req_size) {
  block_info* free_block;

  free_block = FREE_LIST_HEAD;
  while (free_block != NULL) {
    if (SIZE(free_block->size_and_tags) >= req_size) {
      return free_block;
    } else {
      free_block = free_block->next;
    }
  }
  return NULL;
}


/* Insert free_block at the head of the list (LIFO). */
static void insert_free_block(block_info* free_block) {
  block_info* old_head = FREE_LIST_HEAD;
  free_block->next = old_head;
  if (old_head != NULL) {
    old_head->prev = free_block;
  }
  free_block->prev = NULL;
  FREE_LIST_HEAD = free_block;
}


/* Remove a free block from the free list. */
static void remove_free_block(block_info* free_block) {
  block_info* next_free;
  block_info* prev_free;

  next_free = free_block->next;
  prev_free = free_block->prev;

  // If the next block is not null, patch its prev pointer.
  if (next_free != NULL) {
    next_free->prev = prev_free;
  }

  // If we're removing the head of the free list, set the head to be
  // the next block, otherwise patch the previous block's next pointer.
  if (free_block == FREE_LIST_HEAD) {
    FREE_LIST_HEAD = next_free;
  } else {
    prev_free->next = next_free;
  }
}


/* Coalesce 'old_block' with any preceding or following free blocks. */
static void coalesce_free_block(block_info* old_block) {
   fprintf(stderr, "coalescing\n");
  examine_heap();
  block_info* block_cursor;
  block_info* new_block;
  block_info* free_block;
  // size of old block
  size_t old_size = SIZE(old_block->size_and_tags);
  // running sum to be size of final coalesced block
  size_t new_size = old_size;

  // Coalesce with any preceding free block
  block_cursor = old_block;
  while ((block_cursor->size_and_tags & TAG_PRECEDING_USED) == 0) {
    fprintf(stderr, "coalescing with any preceding free block\n");
    // While the block preceding this one in memory (not the
    // prev. block in the free list) is free:

    // Get the size of the previous block from its boundary tag.
    size_t size = SIZE(*((size_t*) UNSCALED_POINTER_SUB(block_cursor, WORD_SIZE)));
    // Use this size to find the block info for that block.
    free_block = (block_info*) UNSCALED_POINTER_SUB(block_cursor, size);
    // Remove that block from free list.
    remove_free_block(free_block);

    // Count that block's size and update the current block pointer.
    new_size += size;
    block_cursor = free_block;
  }
  new_block = block_cursor;


  
  // Coalesce with any following free block.
  // Start with the block following this one in memory
  block_cursor = (block_info*) UNSCALED_POINTER_ADD(old_block, old_size);
  while ((block_cursor->size_and_tags & TAG_USED) == 0) {
    fprintf(stderr, "coalescing with any following free block\n");
    examine_heap();
    // While following block is free:

    size_t size = SIZE(block_cursor->size_and_tags);
    // Remove it from the free list.
    remove_free_block(block_cursor);
    // Count its size and step to the following block.
    new_size += size;
    block_cursor = (block_info*) UNSCALED_POINTER_ADD(block_cursor, size);
  }

   
  // If the block actually grew, remove the old entry from the free-list
  // and add the new entry.
  if (new_size != old_size) {
    fprintf(stderr, "coalesced, removing any old blocks\n");
    examine_heap();
    // Remove the original block from the free list
    remove_free_block(old_block);

    // Save the new size in the block info and in the boundary tag
    // and tag it to show the preceding block is used (otherwise, it
    // would have become part of this one!).
    new_block->size_and_tags = new_size | TAG_PRECEDING_USED;
    // The boundary tag of the preceding block is the word immediately
    // preceding block in memory where we left off advancing block_cursor.
    *(size_t*) UNSCALED_POINTER_SUB(block_cursor, WORD_SIZE) = new_size | TAG_PRECEDING_USED;

    // Put the new block in the free list.
    insert_free_block(new_block);
  }

  fprintf(stderr, "done coalescing\n");
  examine_heap();
  return;
}


/* Get more heap space of size at least req_size. */
static void request_more_space(size_t req_size) {
  size_t pagesize = mem_pagesize();
  size_t num_pages = (req_size + pagesize - 1) / pagesize;
  block_info* new_block;
  size_t total_size = num_pages * pagesize;
  size_t prev_last_word_mask;

  void* mem_sbrk_result = mem_sbrk(total_size);
  if ((size_t) mem_sbrk_result == -1) {
    printf("ERROR: mem_sbrk failed in request_more_space\n");
    exit(0);
  }
  new_block = (block_info*) UNSCALED_POINTER_SUB(mem_sbrk_result, WORD_SIZE);

  // Initialize header by inheriting TAG_PRECEDING_USED status from the
  // end-of-heap word and resetting the TAG_USED bit.
  prev_last_word_mask = new_block->size_and_tags & TAG_PRECEDING_USED;
  new_block->size_and_tags = total_size | prev_last_word_mask;
  // Initialize new footer
  ((block_info*) UNSCALED_POINTER_ADD(new_block, total_size - WORD_SIZE))->size_and_tags =
          total_size | prev_last_word_mask;

  // Initialize new end-of-heap word: SIZE is 0, TAG_PRECEDING_USED is 0,
  // TAG_USED is 1. This trick lets us do the "normal" check even at the end
  // of the heap.
  *((size_t*) UNSCALED_POINTER_ADD(new_block, total_size)) = TAG_USED;

  // Add the new block to the free list and immediately coalesce newly
  // allocated memory space.
  insert_free_block(new_block);
  coalesce_free_block(new_block);
}


/* Initialize the allocator. */
int mm_init() {
  // Head of the free list.
  block_info* first_free_block;

  // Initial heap size: WORD_SIZE byte heap-header (stores pointer to head
  // of free list), MIN_BLOCK_SIZE bytes of space, WORD_SIZE byte heap-footer.
  size_t init_size = WORD_SIZE + MIN_BLOCK_SIZE + WORD_SIZE;
  size_t total_size;

  void* mem_sbrk_result = mem_sbrk(init_size);
  //  printf("mem_sbrk returned %p\n", mem_sbrk_result);
  if ((ssize_t) mem_sbrk_result == -1) {
    printf("ERROR: mem_sbrk failed in mm_init, returning %p\n",
           mem_sbrk_result);
    exit(1);
  }

  first_free_block = (block_info*) UNSCALED_POINTER_ADD(mem_heap_lo(), WORD_SIZE);

  // Total usable size is full size minus heap-header and heap-footer words.
  // NOTE: These are different than the "header" and "footer" of a block!
  //  - The heap-header is a pointer to the first free block in the free list.
  //  - The heap-footer is the end-of-heap indicator (used block with size 0).
  total_size = init_size - WORD_SIZE - WORD_SIZE;

  // The heap starts with one free block, which we initialize now.
  first_free_block->size_and_tags = total_size | TAG_PRECEDING_USED;
  first_free_block->next = NULL;
  first_free_block->prev = NULL;
  // Set the free block's footer.
  *((size_t*) UNSCALED_POINTER_ADD(first_free_block, total_size - WORD_SIZE)) =
	  total_size | TAG_PRECEDING_USED;

  // Tag the end-of-heap word at the end of heap as used.
  *((size_t*) UNSCALED_POINTER_SUB(mem_heap_hi(), WORD_SIZE - 1)) = TAG_USED;

  // Set the head of the free list to this new free block.
  FREE_LIST_HEAD = first_free_block;
  return 0;
}


// TOP-LEVEL ALLOCATOR INTERFACE ------------------------------------

/*
 * Allocate a block of size size and return a pointer to it. If size is zero,
 * returns NULL.
 */
void* mm_malloc(size_t size) {
  size_t req_size;
  block_info* ptr_free_block = NULL;
  size_t block_size;
  size_t preceding_block_use_tag;


  examine_heap();

  // Zero-size requests get NULL.
  if (size == 0) {
    return NULL;
  }

  // Add one word for the initial size header.
  // Note that we don't need a footer when the block is used/allocated!
  size += WORD_SIZE; //word size is 8B, size of header
  if (size <= MIN_BLOCK_SIZE) {
    // Make sure we allocate enough space for the minimum block size.
    req_size = MIN_BLOCK_SIZE;
  } else {
    // Round up for proper alignment.
    req_size = ALIGNMENT * ((size + ALIGNMENT - 1) / ALIGNMENT);
  }

  // TODO: Implement mm_malloc.  You can change or remove any of the
  // above code.  It is included as a suggestion of where to start.
  // You will want to replace this return statement...

 fprintf(stderr, "called to allocate %d BYTES, allocating req size of %d BYTES \n", size-WORD_SIZE, req_size); //have to print to standard error to show in debugging



 ptr_free_block = search_free_list(req_size); //sets [ptr_free_block] to the pointer of a free block
 //search free list returns a pointer to a blockinfo struct
 //should have tags: current used, prev used, pointers to next, and previous block

  if (ptr_free_block == NULL) {
    // No suitable block found, request more space.
    fprintf(stderr, "REQUESTING MORE SPACE \n");
     fprintf(stderr, "current size of heap %d bytes \n", mem_heapsize());
    
    request_more_space(req_size); //throws errpr when we request 56 bytes???? , seg fault
    //maybe since the allocated block is not in free list?
    //error when freeblock->next is called

     fprintf(stderr, "done requesting space \n");

    ptr_free_block = search_free_list(req_size);
  }
  
 

  fprintf(stderr, "FOUND FREE BLOCK OF %d bytes big\n", SIZE(ptr_free_block->size_and_tags) );


  // Remove the block we found from the free list.
  remove_free_block(ptr_free_block);

  // Check if we need to split the block.
    //uses the Size() macro to extract the ptr_free_block.size_and_tags
  //uses the -> to get the size_and_tags size_t info to use in the SIZE macro
  block_size = SIZE(ptr_free_block->size_and_tags);


  //if the size of the block is greater than the required size, and the min block size
  //in the case we use a really big free block, we should split the part we dont need to not use all free space in heap
  //if the block size of 

  if (block_size - req_size >= MIN_BLOCK_SIZE) {
    //ONLY SPLITS IF THE SPLITTED FREE BLOCK WOULD FIT THE MINIMUM 32BYTE BLOCK SIZE, 8B HEAD, 8B NEXT, 8B PREV, 8BFOOTER
    //WE WERE GETTING ERROR BECAUSE THE COALESNE FUNCTION WAS TRYING TO ACCESS .NEXT ON 8B BLOCK THAT COULDN'T HAVE A .NEXT
    //ACCSESSING OUT OF BOUNDS CAUSED A SEG FAULT

    fprintf(stderr, "splitting block since free block was too big\n");

    //Split the block.
    block_info* remaining_block = (block_info*)UNSCALED_POINTER_ADD(ptr_free_block, req_size);  //create a new block pointing to the start of the extra space we will split

    //update size of block to free
    remaining_block->size_and_tags = block_size - req_size; //the size of our block - the size we wanted to allocate

   //change the preceding used tag to 1 for new block, since comes after allocated blocks 
    remaining_block->size_and_tags |= TAG_PRECEDING_USED; //sets the preceding bit used to 1 in the following block

     // Update the remaining block's footer.
    ((block_info*)UNSCALED_POINTER_ADD(remaining_block, SIZE(remaining_block->size_and_tags) - WORD_SIZE))->size_and_tags = remaining_block->size_and_tags;


    //Update the size of the allocated block.
    //bitwise or combines the required size and previous preceding bit and tag used bit (now equal to 1)
    // |tag_used = 1 is a mask for the LSB, when used like this it sets the current tag_used = 1
    ptr_free_block->size_and_tags = (req_size | TAG_PRECEDING_USED) | TAG_USED ;
 

    fprintf(stderr, "split block is %d bytes big\n", SIZE(remaining_block->size_and_tags) );


    
    // Insert the remaining block back into the free list.
    insert_free_block(remaining_block);
  } 
  
  else {
    // Use the entire block.
    fprintf(stderr, "not splitting, perfect size free block found\n");
    ptr_free_block->size_and_tags |= TAG_USED; //sets the used bit
    
  }

  fprintf(stderr, "allocated %d, req size is %d size \n", size-WORD_SIZE, req_size); //have to print to standard error to show in debugging
  fprintf(stderr, "examining heap \n"); //have to print to standard error to show in debugging
  examine_heap(); //prints current heap, only prints free objects
 

  // Return a pointer to the payload of the allocated block.
  return (void*)UNSCALED_POINTER_ADD(ptr_free_block, WORD_SIZE);
  //moves the pointer ahead 8B since the pointer points to the header, when we want it to return a pointer pointing to the payload
  //then casts the returned value of the unscaled pointer add back to a void (since it was cast to a char for the macro)

  //allocates a block of size _

  /*
  Figure out how big a block you need (factor in alignment, minimum block size, etc)
2. Call search_free_list() to get a free block that is large enough
a. NOTE: this will yield a block that is AT LEAST the request size
b. What happens if we run out of space in the heap?
3. Remove that block from the free list
a. May need to split this block to prevent excessive internal fragmentation (see
NOTE above) -- what dictates whether we can split?
b. May need to reinsert extra block into the free list if we split
4. Update size_and_tags appropriately (do preceding and following blocks need
updating?)
5. Return a pointer to the payload of that block
*/


}


/* Free the block referenced by ptr. */
void mm_free(void* ptr) {
  size_t payload_size;
  block_info* block_to_free;
  block_info* following_block;
  block_info* footer;
  size_t block_size;

  // TODO: Implement mm_free.  You can change or remove the declaraions
  // above.  They are included as minor hints.

  if (ptr == NULL) {
        return;
    }


  //have to test if previous block is allocated somehow
  //or maybe test if head dont set previous block to be allocated


  // Convert the given used block into a free block.
  //subtracts the pointer to the block by the word size, given allocated block will point to payload so we need to move back 8bytes to get to start of block (header)
  //block to free points to start of block
  block_to_free = (block_info*)UNSCALED_POINTER_SUB(ptr, WORD_SIZE);

  block_size = SIZE(block_to_free->size_and_tags);

  fprintf(stderr, "FREE CALLED for %d bytes \n", SIZE(block_to_free->size_and_tags));


  //sets block to free tag used bit to 0
  block_to_free->size_and_tags &= ~TAG_USED; //keeps everything but sets tag used bit to zero for the block we are going to free


  //maybe jump to prev block and check if it is allocated?, but how we dont now size?


   //following block starts at end of current block
  following_block = (block_info*)UNSCALED_POINTER_ADD(block_to_free, block_size);

  //jump to following block and change the preceding used tag to 0
  following_block->size_and_tags &= ~TAG_PRECEDING_USED; //sets the preceding bit used to 0 in the following block



    //update the footer size_and_tags for block to free
     //block to free + size - word sizes = start of footer
  //sets the size and tags to be the same as the header
  footer =  ((block_info*)UNSCALED_POINTER_SUB(following_block, WORD_SIZE));
  footer->size_and_tags = block_to_free->size_and_tags;


    //reinsert the free block into the head of the free list.
   insert_free_block(block_to_free);


    //coalesce preceding and following blocks if necessary.
  coalesce_free_block(block_to_free);



  examine_heap();

  //if we free a block we need to change the block afters bits of preceding used to 0

}


/*
 * A heap consistency checker. Optional, but recommended to help you debug
 * potential issues with your allocator.
 */
int mm_check() {
  // TODO: Implement a heap consistency checker as needed/desired.
  return 0;
}