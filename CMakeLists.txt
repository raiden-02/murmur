cmake_minimum_required(VERSION 3.25)

project(murmur
    VERSION 0.1.0
    DESCRIPTION "A cyberpunk night market whose NPCs think on a from-scratch GPU inference engine"
    LANGUAGES C CXX
)

# Host C++ is C++23. CUDA (.cu) files stay C++20: nvcc can't use C++23 with
# MSVC as its host compiler yet, so headers shared with .cu files must be C++20.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# raylib: window, input and 2D rendering for the game.
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    raylib
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    URL https://github.com/raysan5/raylib/archive/refs/tags/6.0.tar.gz
    URL_HASH SHA256=2b3ee1e2120c7a0796b33062c7e9a694dd8a8caa56a96319ac8c8ecf54a90d0b
)
FetchContent_MakeAvailable(raylib)

# nlohmann/json: parses config.json, tokenizer.json and safetensors headers.
FetchContent_Declare(
    nlohmann_json
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
)
FetchContent_MakeAvailable(nlohmann_json)

# Model files on the host: safetensors weights, config.json and the tokenizer.
# Plain C++ with no CUDA dependency.
add_library(murmur-model STATIC
    src/model/mapped_file.cpp
    src/model/safetensors.cpp
    src/model/model_config.cpp
    src/model/model_files.cpp
    src/model/unicode.cpp
    src/model/unicode_tables.cpp
    src/model/pre_tokenizer.cpp
    src/model/tokenizer.cpp
    src/model/chat_template.cpp
)
target_include_directories(murmur-model PUBLIC src)
target_link_libraries(murmur-model PRIVATE nlohmann_json::nlohmann_json)

# Command-line flags with --help and errors for unknown ones: the lab's
# commands and the game read their arguments with it.
add_library(murmur-args STATIC
    src/lab/args.cpp
)
target_include_directories(murmur-args PUBLIC src)

# Unit tests (GoogleTest). The host tests need no GPU and build without
# CUDA: Unicode, the tokenizer, safetensors, config.json, the model folders
# and argument parsing. Tests that need the downloaded model or the
# reference dumps find them through MURMUR_SOURCE_DIR (or $MURMUR_MODEL_DIR),
# and skip themselves when the files are absent.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    URL_HASH SHA256=40d4ec942217dcc84a9ebe2a68584ada7d4a33a8ee958755763278ea1c5e18ff
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

add_executable(murmur-host-tests
    tests/unicode_tests.cpp
    tests/pre_tokenizer_tests.cpp
    tests/tokenizer_tests.cpp
    tests/safetensors_tests.cpp
    tests/model_config_tests.cpp
    tests/model_files_tests.cpp
    tests/args_tests.cpp
)
target_link_libraries(murmur-host-tests PRIVATE murmur-model murmur-args nlohmann_json::nlohmann_json
                                                GTest::gtest_main)
target_compile_definitions(murmur-host-tests PRIVATE MURMUR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(murmur-host-tests)

# CUDA targets are added only when a CUDA compiler is available.
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    # RTX 4080 SUPER is Ada Lovelace, compute capability 8.9.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Gives plain .cpp files the CUDA headers and runtime library too.
    find_package(CUDAToolkit REQUIRED)

    # Shared CUDA helpers: error checking and device memory ownership.
    # Linking it also provides the src/ include path and the CUDA runtime.
    add_library(murmur-common STATIC
        src/common/cuda_check.cpp
        src/common/cuda_graph.cpp
        src/common/device_info.cpp
        src/common/cuda_stream.cpp
        src/common/gpu_timer.cpp
    )
    target_include_directories(murmur-common PUBLIC src)
    target_link_libraries(murmur-common PUBLIC CUDA::cudart)

    # GPU kernels and the host functions that launch them.
    add_library(murmur-kernels STATIC
        src/kernels/vector_add.cu
        src/kernels/reduce_sum.cu
        src/kernels/transpose.cu
        src/kernels/matmul.cu
    )
    target_link_libraries(murmur-kernels PUBLIC murmur-common CUDA::cublas)

    # The inference engine: GPU kernels for the transformer forward pass and
    # the host code that drives them.
    add_library(murmur-engine STATIC
        src/engine/kernels/elementwise.cu
        src/engine/kernels/rms_norm.cu
        src/engine/kernels/rope.cu
        src/engine/kernels/attention.cu
        src/engine/kernels/decode_attention.cu
        src/engine/kernels/argmax.cu
        src/engine/kernels/matvec.cu
        src/engine/kernels/gemm.cu
        src/engine/kernels/quantize.cu
        src/engine/kernels/sampling.cu
        src/engine/blas.cpp
        src/engine/weight_matrix.cpp
        src/engine/channel_importance.cpp
        src/engine/linear.cpp
        src/engine/model_weights.cpp
        src/engine/kv_cache.cpp
        src/engine/page_pool.cpp
        src/engine/prefix_cache.cpp
        src/engine/kv_cache_manager.cpp
        src/engine/transformer.cpp
        src/engine/sampler.cpp
        src/engine/gpu_sampler.cpp
        src/engine/chat_session.cpp
        src/engine/grammar.cpp
        src/engine/speculator.cpp
        src/engine/engine.cpp
        src/engine/model_loader.cpp
        src/engine/inference_service.cpp
    )
    target_link_libraries(murmur-engine PUBLIC murmur-common murmur-model CUDA::cublas CUDA::cublasLt)

    # The market's action language: shared by the game, the lab and the tests.
    add_library(murmur-market STATIC
        src/game/action_language.cpp
    )
    target_link_libraries(murmur-market PUBLIC murmur-common)

    # The game: the night market, with NPC dialogue generated by the engine.
    # Everything but main() is a library, so the tests can run the market's
    # bookkeeping and the text layout without a window.
    add_library(murmur-game STATIC
        src/game/street.cpp
        src/game/post_fx.cpp
        src/game/text_layout.cpp
        src/game/dialogue.cpp
        src/game/market.cpp
        src/game/hud.cpp
        src/game/demo.cpp
    )
    target_link_libraries(murmur-game PUBLIC murmur-engine murmur-market raylib)
    add_executable(murmur
        src/game/main.cpp
    )
    target_link_libraries(murmur PRIVATE murmur-game murmur-args)

    # Lab: `murmur-lab <command>` runs GPU experiments and benchmarks.
    add_executable(murmur-lab
        src/lab/lab_main.cpp
        src/lab/device_query.cu
        src/lab/vector_add_benchmark.cpp
        src/lab/reduce_sum_benchmark.cpp
        src/lab/transpose_benchmark.cpp
        src/lab/matmul_benchmark.cpp
        src/lab/chat_commands.cpp
        src/lab/decode_benchmark.cpp
        src/lab/matvec_benchmark.cpp
        src/lab/batch_benchmark.cpp
        src/lab/market_benchmark.cpp
        src/lab/schedule_trace.cpp
        src/lab/quality.cpp
        src/lab/speculative_benchmark.cpp
        src/lab/actions_benchmark.cpp
        src/lab/compare_benchmark.cpp
        src/lab/attention_benchmark.cpp
        src/lab/logit_comparison.cu
    )
    target_link_libraries(murmur-lab PRIVATE murmur-kernels murmur-engine murmur-market murmur-args
                                         nlohmann_json::nlohmann_json)

    # The tests that run on the GPU, or on the engine and the game.
    add_executable(murmur-tests
        tests/device_buffer_tests.cpp
        tests/vector_add_tests.cpp
        tests/reduce_sum_tests.cpp
        tests/transpose_tests.cpp
        tests/matmul_tests.cpp
        tests/engine_kernels_tests.cpp
        tests/transformer_tests.cpp
        tests/sampler_tests.cpp
        tests/inference_service_tests.cpp
        tests/kv_paging_tests.cpp
        tests/gpu_sampler_tests.cpp
        tests/engine_tests.cpp
        tests/quantization_tests.cpp
        tests/draft_model_tests.cpp
        tests/speculative_tests.cpp
        tests/grammar_tests.cpp
        tests/action_language_tests.cpp
        tests/market_tests.cpp
        tests/text_layout_tests.cpp
    )
    target_link_libraries(murmur-tests PRIVATE murmur-kernels murmur-engine murmur-market murmur-game
                                               nlohmann_json::nlohmann_json GTest::gtest_main)
    target_compile_definitions(murmur-tests PRIVATE MURMUR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    gtest_discover_tests(murmur-tests)
else()
    message(STATUS "CUDA compiler not found: only the host-side libraries and their tests are built. "
                   "Install the CUDA Toolkit to build the game, the lab and the GPU tests.")
endif()

# Every target of ours (not the fetched dependencies) compiles at warning
# level 4 with warnings as errors, so a new warning is fixed when it appears.
# CUDA files pass the same flags to MSVC compiling their host code, minus
# C4211, which fires on the launch stubs nvcc itself generates for kernels in
# an anonymous namespace (code we can't change).
if(MSVC)
    foreach(target murmur-model murmur-args murmur-host-tests murmur-common murmur-kernels murmur-engine
                   murmur-market murmur-game murmur murmur-lab murmur-tests)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive- /utf-8>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/WX,/wd4211,/utf-8>)
        endif()
    endforeach()
endif()
