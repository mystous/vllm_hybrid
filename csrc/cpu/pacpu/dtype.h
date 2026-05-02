#if defined(__x86_64__) && !defined(ISPC)
  typedef _Float16 __fp16;
#endif
#ifdef ISPC
  typedef float16 data_t;
#else
  typedef __fp16 data_t;
#endif
// #if defined(ISPC_TARGET_AVX2) || defined(__AVX2__)
//   typedef float itmd_t;
// #else
//   typedef data_t itmd_t;
// #endif
typedef float itmd_t;
typedef float otpt_t;

#define GEMM cblas_hgemm
#define HEAD_DIM 128 // Constant for all models
#define BLOCK_SIZE 16

#if defined(LLAMA3_8B)
  #define NUM_LAYERS 32
  #define NUM_Q_HEADS (32 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(LLAMA2_7B)
  #define NUM_LAYERS 32
  #define NUM_Q_HEADS (32 / TP_DEGREE)
  #define NUM_KV_HEADS (32 / TP_DEGREE)
#elif defined(LLAMA2_13B)  
  #define NUM_LAYERS 40
  #define NUM_Q_HEADS (40 / TP_DEGREE)
  #define NUM_KV_HEADS (40 / TP_DEGREE)
#elif defined(LLAMA2_70B) || defined(LLAMA3_70B) || defined(LLAMA3_3_70B)
  // Llama-3.3-70B = Llama-3 70B layout (80 layers, 64 Q heads, 8 KV heads).
  // IDE_006 prod target. Added for vLLM integration (TSK_018).
  #define NUM_LAYERS 80
  #define NUM_Q_HEADS (64 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(QWEN2_5_1_5B)
  // IDE_006 dev target. Qwen2.5-1.5B-Instruct: 28 layers, 12 Q heads, 2 KV heads.
  // Added for vLLM integration (TSK_018).
  #define NUM_LAYERS 28
  #define NUM_Q_HEADS (12 / TP_DEGREE)
  #define NUM_KV_HEADS (2 / TP_DEGREE)
#elif defined(QWEN2_5_7B)
  // Qwen2.5-7B-Instruct: 28 layers, 28 Q heads, 4 KV heads (head_dim=128).
  #define NUM_LAYERS 28
  #define NUM_Q_HEADS (28 / TP_DEGREE)
  #define NUM_KV_HEADS (4 / TP_DEGREE)
#elif defined(QWEN2_5_32B)
  // Qwen2.5-32B-Instruct: 64 layers, 40 Q heads, 8 KV heads (head_dim=128).
  #define NUM_LAYERS 64
  #define NUM_Q_HEADS (40 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(QWEN2_5_72B)
  // Qwen2.5-72B-Instruct: 80 layers, 64 Q heads, 8 KV heads (head_dim=128).
  #define NUM_LAYERS 80
  #define NUM_Q_HEADS (64 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(MISTRAL_7B)
  // Mistral-7B-Instruct-v0.3 (head_dim=128): 32 layers, 32 Q heads, 8 KV heads.
  #define NUM_LAYERS 32
  #define NUM_Q_HEADS (32 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(MISTRAL_NEMO_12B)
  // Mistral-Nemo-Instruct-2407 (head_dim=128): 40 layers, 32 Q heads, 8 KV heads.
  #define NUM_LAYERS 40
  #define NUM_Q_HEADS (32 / TP_DEGREE)
  #define NUM_KV_HEADS (8 / TP_DEGREE)
#elif defined(PHI3_MEDIUM_14B)
  // Phi-3-medium-128k-instruct (head_dim=128): 40 layers, 40 Q heads, 10 KV heads.
  #define NUM_LAYERS 40
  #define NUM_Q_HEADS (40 / TP_DEGREE)
  #define NUM_KV_HEADS (10 / TP_DEGREE)
#else
  #error "Please define the model (e.g. -DLLAMA3_3_70B / -DQWEN2_5_1_5B / -DQWEN2_5_7B / -DQWEN2_5_32B / -DQWEN2_5_72B / -DMISTRAL_7B / -DMISTRAL_NEMO_12B / -DPHI3_MEDIUM_14B)"
#endif

#define QH_PER_KVH (NUM_Q_HEADS / NUM_KV_HEADS)
#define BLOCK_NELEM (NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM)

#define MAX_BATCH_SIZE 4096
#define MAX_WS 256
#define MAX_TOK_NUM 1048576 // Maxinum number of token's KV to be scanned in one iteration
#define MAX_TASK_NUM (MAX_BATCH_SIZE + MAX_WS)