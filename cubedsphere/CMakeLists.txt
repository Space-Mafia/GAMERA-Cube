cmake_minimum_required(VERSION 3.16)
project(gamera_cubedsphere C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
enable_testing()
option(GAMERA_REQUIRE_PRODUCTION "Require MPI and parallel HDF5 Earth driver" OFF)
find_package(OpenMP)
find_package(Python3 COMPONENTS Interpreter)
find_package(MPI COMPONENTS C)
if(MPI_C_FOUND)
    set(HDF5_PREFER_PARALLEL TRUE)
    find_package(HDF5 COMPONENTS C QUIET)
endif()
if(GAMERA_REQUIRE_PRODUCTION AND NOT (MPI_C_FOUND AND HDF5_FOUND AND HDF5_IS_PARALLEL))
    message(FATAL_ERROR "Production requires MPI C and parallel HDF5 from the same MPI installation.")
endif()

set(NO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../nonorthogonal/src/nonorthogonal)
set(NO_APP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../nonorthogonal/src)
add_library(gamera_cs_mhd STATIC
    ${NO_APP_SRC}/nonorthogonal_radial_map.c
    ${NO_APP_SRC}/solar_wind.c
    ${NO_SRC}/nonorthogonal_advance.c
    ${NO_SRC}/nonorthogonal_background.c
    ${NO_SRC}/nonorthogonal_flux.c
    ${NO_SRC}/nonorthogonal_geometry.c
    ${NO_SRC}/nonorthogonal_grid.c
    ${NO_SRC}/nonorthogonal_initialization.c
    ${NO_SRC}/nonorthogonal_mesh.c
    ${NO_SRC}/nonorthogonal_mi.c
    ${NO_SRC}/nonorthogonal_operators.c
    ${NO_SRC}/nonorthogonal_reconstruction.c
    ${NO_SRC}/nonorthogonal_state.c
    ${NO_SRC}/nonorthogonal_step.c
    ${NO_SRC}/nonorthogonal_storage.c
    ${NO_SRC}/nonorthogonal_sweep.c
    cubedsphere_halo.c
    cubedsphere_interface.c
    cubedsphere_step.c
    cubedsphere_recovery.c
    cubedsphere_earth_startup.c
    cubedsphere_earth_boundary.c
    cubedsphere_mi_dipole.c
    cubedsphere_mesh.c
    cubedsphere_decomposition.c)
set_source_files_properties(${NO_SRC}/nonorthogonal_operators.c PROPERTIES
    COMPILE_DEFINITIONS
    "gamera_no_flux_to_cell_field=gamera_cs_uncalibrated_flux_to_cell_field")
target_include_directories(gamera_cs_mhd PUBLIC
    ${NO_SRC} ${NO_APP_SRC} ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_C_FOUND)
    target_link_libraries(gamera_cs_mhd PUBLIC OpenMP::OpenMP_C)
endif()
target_link_libraries(gamera_cs_mhd PUBLIC m)
add_executable(test_earth_interior_recovery tests/test_earth_interior_recovery.c)
target_link_libraries(test_earth_interior_recovery PRIVATE gamera_cs_mhd)
add_test(NAME earth_interior_recovery COMMAND test_earth_interior_recovery)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(gamera_cs_mhd PRIVATE -fno-fast-math)
endif()

add_executable(test_cubedsphere_mesh tests/test_cubedsphere_mesh.c)
target_link_libraries(test_cubedsphere_mesh PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_mesh COMMAND test_cubedsphere_mesh)

add_executable(test_cubedsphere_decomposition
    tests/test_cubedsphere_decomposition.c)
target_link_libraries(test_cubedsphere_decomposition PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_decomposition COMMAND test_cubedsphere_decomposition)
add_executable(test_cubedsphere_halo_bounded tests/test_cubedsphere_halo_bounded.c)
target_link_libraries(test_cubedsphere_halo_bounded PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_halo_bounded COMMAND test_cubedsphere_halo_bounded)
add_executable(test_cubedsphere_halo_fastpath tests/test_cubedsphere_halo_fastpath.c)
target_link_libraries(test_cubedsphere_halo_fastpath PRIVATE gamera_cs_mhd)
target_compile_options(test_cubedsphere_halo_fastpath PRIVATE -fno-fast-math)
add_test(NAME cubedsphere_halo_fastpath COMMAND test_cubedsphere_halo_fastpath)
add_executable(test_cubedsphere_emf_cache tests/test_cubedsphere_emf_cache.c)
target_link_libraries(test_cubedsphere_emf_cache PRIVATE gamera_cs_mhd)
target_compile_options(test_cubedsphere_emf_cache PRIVATE -fno-fast-math)
add_test(NAME cubedsphere_emf_cache COMMAND test_cubedsphere_emf_cache)
set_tests_properties(cubedsphere_emf_cache PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
add_test(NAME cubedsphere_emf_cache_omp2 COMMAND test_cubedsphere_emf_cache)
set_tests_properties(cubedsphere_emf_cache_omp2 PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
add_executable(test_cubedsphere_primitive_cache tests/test_cubedsphere_primitive_cache.c)
target_link_libraries(test_cubedsphere_primitive_cache PRIVATE gamera_cs_mhd)
target_compile_options(test_cubedsphere_primitive_cache PRIVATE -fno-fast-math)
add_test(NAME cubedsphere_primitive_cache COMMAND test_cubedsphere_primitive_cache)
set_tests_properties(cubedsphere_primitive_cache PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
add_test(NAME cubedsphere_primitive_cache_omp2 COMMAND test_cubedsphere_primitive_cache)
set_tests_properties(cubedsphere_primitive_cache_omp2 PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")

add_executable(test_cubedsphere_earth_radial_map
    tests/test_cubedsphere_earth_radial_map.c)
target_link_libraries(test_cubedsphere_earth_radial_map PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_earth_radial_map
    COMMAND test_cubedsphere_earth_radial_map)

add_executable(export_cubedsphere_grid
    tests/export_cubedsphere_grid.c)
target_link_libraries(export_cubedsphere_grid PRIVATE gamera_cs_mhd)

add_executable(test_cubedsphere_uniform_mhd tests/test_cubedsphere_uniform_mhd.c)
target_link_libraries(test_cubedsphere_uniform_mhd PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_uniform_mhd COMMAND test_cubedsphere_uniform_mhd)

add_executable(test_cubedsphere_six_mhd tests/test_cubedsphere_six_mhd.c)
target_link_libraries(test_cubedsphere_six_mhd PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_six_mhd COMMAND test_cubedsphere_six_mhd)

add_executable(test_cubedsphere_dipole_background
    tests/test_cubedsphere_dipole_background.c)
target_link_libraries(test_cubedsphere_dipole_background PRIVATE gamera_cs_mhd)
add_executable(test_cubedsphere_mi_shell
    tests/test_cubedsphere_mi_shell.c)
target_link_libraries(test_cubedsphere_mi_shell PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_mi_three_layer_shell_parity
    COMMAND test_cubedsphere_mi_shell)
add_test(NAME cubedsphere_dipole_background
    COMMAND test_cubedsphere_dipole_background)
add_test(NAME cubedsphere_dipole_background_72x16x16
    COMMAND test_cubedsphere_dipole_background earth-local)
add_test(NAME cubedsphere_earth_mi_dipole_zero_fac
    COMMAND test_cubedsphere_dipole_background earth-mi-local 1)
if(Python3_Interpreter_FOUND)
    add_test(NAME cubedsphere_earth_drive_shape_inference
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_earth_drive_output_shape.py)
    add_test(NAME cubedsphere_earth_mi_dipole_single_fluid_reduction
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_earth_mi_reduction.py
        $<TARGET_FILE:test_cubedsphere_dipole_background>)
endif()

add_executable(test_cubedsphere_mhd_blast tests/test_cubedsphere_mhd_blast.c)
target_link_libraries(test_cubedsphere_mhd_blast PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_mhd_blast COMMAND test_cubedsphere_mhd_blast)

add_executable(test_cubedsphere_hd_spherical_blast
    tests/test_cubedsphere_hd_spherical_blast.c)
target_link_libraries(test_cubedsphere_hd_spherical_blast PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_hd_spherical_blast
    COMMAND test_cubedsphere_hd_spherical_blast)
add_test(NAME cubedsphere_hd_spherical_blast_six_faces
    COMMAND test_cubedsphere_hd_spherical_blast - 24 1.5)
add_test(NAME cubedsphere_mhd_spherical_blast_face_center
    COMMAND test_cubedsphere_hd_spherical_blast - 32 1.5 face-center-mhd 0.1)

add_executable(test_cubedsphere_mhd_shock tests/test_cubedsphere_mhd_shock.c)
target_link_libraries(test_cubedsphere_mhd_shock PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_mhd_shock COMMAND test_cubedsphere_mhd_shock)
add_test(NAME cubedsphere_mhd_shock_300_steps
    COMMAND test_cubedsphere_mhd_shock 300)

add_executable(test_cubedsphere_alfven tests/test_cubedsphere_alfven.c)
target_link_libraries(test_cubedsphere_alfven PRIVATE gamera_cs_mhd)
add_test(NAME cubedsphere_alfven COMMAND test_cubedsphere_alfven)
add_test(NAME cubedsphere_alfven_200_steps
    COMMAND test_cubedsphere_alfven 9 200 0.002)
if(Python3_Interpreter_FOUND)
    add_test(NAME cubedsphere_alfven_initial_field_order
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_alfven_initial_order.py
            $<TARGET_FILE:test_cubedsphere_alfven>)
    add_test(NAME cubedsphere_alfven_evolution_order
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_alfven_evolution_order.py
            $<TARGET_FILE:test_cubedsphere_alfven>)
endif()

if(MPI_C_FOUND)
    # Reuse the accepted Yin--Yang sparse donor exchange implementation.
    add_library(gamera_cs_sparse_plan STATIC
        ${NO_APP_SRC}/nonorthogonal_backend/nonorthogonal_yinyang_sparse_plan.c)
    target_compile_definitions(gamera_cs_sparse_plan PRIVATE
        GAMERA_YINYANG_SPARSE_OVERSET)
    target_include_directories(gamera_cs_sparse_plan PUBLIC
        ${NO_APP_SRC}/nonorthogonal_backend)
    target_link_libraries(gamera_cs_sparse_plan PUBLIC MPI::MPI_C)
    add_library(gamera_cs_mpi STATIC cubedsphere_mpi.c cubedsphere_mpi_step.c
        cubedsphere_mpi_tile.c cubedsphere_mi_dipole_mpi.c)
    target_link_libraries(gamera_cs_mpi PUBLIC gamera_cs_mhd MPI::MPI_C
        gamera_cs_sparse_plan)
    target_include_directories(gamera_cs_mpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(test_cubedsphere_halo_block tests/test_cubedsphere_halo_block.c)
    target_link_libraries(test_cubedsphere_halo_block PRIVATE gamera_cs_mpi)
    add_executable(test_cubedsphere_halo_parallel tests/test_cubedsphere_halo_parallel.c)
    target_link_libraries(test_cubedsphere_halo_parallel PRIVATE gamera_cs_mpi)
    add_executable(test_cubedsphere_mpi_halo
        tests/test_cubedsphere_mpi_halo.c)
    target_link_libraries(test_cubedsphere_mpi_halo PRIVATE gamera_cs_mpi)
    add_executable(test_cubedsphere_mpi_tile
        tests/test_cubedsphere_mpi_tile.c)
    target_link_libraries(test_cubedsphere_mpi_tile PRIVATE gamera_cs_mpi)
    add_executable(test_cubedsphere_mpi_earth_tile
        tests/test_cubedsphere_mpi_earth_tile.c)
    target_link_libraries(test_cubedsphere_mpi_earth_tile PRIVATE gamera_cs_mpi)
    # The validated physical-tile path also serves as the Earth drive runner;
    # only its driver/output mode differs from the short parity tests.
    if(HDF5_FOUND AND HDF5_IS_PARALLEL)
        add_executable(cubedsphere_earth_mpi_drive
            tests/test_cubedsphere_mpi_earth_tile.c
            cubedsphere_analysis_hdf5.c)
        target_compile_definitions(cubedsphere_earth_mpi_drive PRIVATE
            GAMERA_CS_DRIVE_HDF5=1)
        target_include_directories(cubedsphere_earth_mpi_drive PRIVATE
            ${HDF5_INCLUDE_DIRS})
        target_link_libraries(cubedsphere_earth_mpi_drive PRIVATE
            gamera_cs_mpi ${HDF5_LIBRARIES})
        add_executable(check_cubedsphere_earth_hdf5
            tests/check_cubedsphere_earth_hdf5.c)
        target_include_directories(check_cubedsphere_earth_hdf5 PRIVATE
            ${HDF5_INCLUDE_DIRS})
        target_link_libraries(check_cubedsphere_earth_hdf5 PRIVATE
            ${HDF5_LIBRARIES} m)
    else()
        message(STATUS "Parallel HDF5 unavailable: production Earth drive disabled; MPI parity tests remain available")
    endif()
    add_executable(test_cubedsphere_mi_shell_mpi
        tests/test_cubedsphere_mi_shell_mpi.c)
    target_link_libraries(test_cubedsphere_mi_shell_mpi PRIVATE gamera_cs_mpi)
    add_executable(test_cubedsphere_dipole_background_mpi
        tests/test_cubedsphere_dipole_background.c)
    target_compile_definitions(test_cubedsphere_dipole_background_mpi
        PRIVATE GAMERA_CS_MPI_DRIVER=1)
    target_link_libraries(test_cubedsphere_dipole_background_mpi
        PRIVATE gamera_cs_mpi)
    set(CS_MPIEXEC_EXTRA_FLAGS)
    if(APPLE)
        execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version
            OUTPUT_VARIABLE CS_MPIEXEC_VERSION
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(CS_MPIEXEC_VERSION MATCHES "Open MPI")
            # Homebrew OpenMPI may expose only one slot in sandboxed CTest.
            list(APPEND CS_MPIEXEC_EXTRA_FLAGS --oversubscribe --map-by slot)
        endif()
    endif()
    foreach(layout radial angular)
        if(layout STREQUAL "radial")
            set(block_dims 2 1 1)
        else()
            set(block_dims 1 2 1)
        endif()
        add_test(NAME cubedsphere_halo_block_${layout}
            COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
                ${MPIEXEC_NUMPROC_FLAG} 12
                $<TARGET_FILE:test_cubedsphere_halo_block> ${block_dims})
        set_tests_properties(cubedsphere_halo_block_${layout} PROPERTIES
            TIMEOUT 120 PROCESSORS 12 ENVIRONMENT "OMP_NUM_THREADS=1")
        foreach(threads 1 2)
            add_test(NAME cubedsphere_halo_parallel_${layout}_omp${threads}
                COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
                    ${MPIEXEC_NUMPROC_FLAG} 12
                    $<TARGET_FILE:test_cubedsphere_halo_parallel> ${block_dims})
            math(EXPR block_processors "12 * ${threads}")
            set_tests_properties(cubedsphere_halo_parallel_${layout}_omp${threads}
                PROPERTIES TIMEOUT 120 PROCESSORS ${block_processors}
                ENVIRONMENT "OMP_NUM_THREADS=${threads}")
        endforeach()
    endforeach()
    add_test(NAME cubedsphere_mpi_halo
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 6
            $<TARGET_FILE:test_cubedsphere_mpi_halo>)
    set_tests_properties(cubedsphere_mpi_halo PROPERTIES PROCESSORS 6)
    add_test(NAME cubedsphere_mpi_tile_radial
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mpi_tile> 2 1 1)
    set_tests_properties(cubedsphere_mpi_tile_radial PROPERTIES
        PROCESSORS 12 TIMEOUT 120)
    add_test(NAME cubedsphere_mpi_tile_angular
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mpi_tile> 1 2 1)
    set_tests_properties(cubedsphere_mpi_tile_angular PROPERTIES
        PROCESSORS 12 TIMEOUT 120)
    add_test(NAME cubedsphere_mpi_earth_tile_radial
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mpi_earth_tile>
            2 1 1 16 16 16)
    set_tests_properties(cubedsphere_mpi_earth_tile_radial PROPERTIES
        PROCESSORS 12 TIMEOUT 180)
    add_test(NAME cubedsphere_mpi_earth_tile_angular
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 24
            $<TARGET_FILE:test_cubedsphere_mpi_earth_tile>
            1 2 2 16 16 16)
    set_tests_properties(cubedsphere_mpi_earth_tile_angular PROPERTIES
        PROCESSORS 24 TIMEOUT 180)
    add_test(NAME cubedsphere_mi_shell_mpi_radial
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mi_shell_mpi> 2 1 1)
    set_tests_properties(cubedsphere_mi_shell_mpi_radial PROPERTIES
        PROCESSORS 12 TIMEOUT 180)
    add_test(NAME cubedsphere_mi_shell_mpi_radial_cross
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mi_shell_mpi> 2 1 1 7)
    set_tests_properties(cubedsphere_mi_shell_mpi_radial_cross PROPERTIES
        PROCESSORS 12 TIMEOUT 180)
    add_test(NAME cubedsphere_mi_shell_mpi_angular
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 24
            $<TARGET_FILE:test_cubedsphere_mi_shell_mpi> 1 2 2)
    set_tests_properties(cubedsphere_mi_shell_mpi_angular PROPERTIES
        PROCESSORS 24 TIMEOUT 180)
    add_test(NAME cubedsphere_mi_earth_tile_radial
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mpi_earth_tile>
            2 1 1 72 16 16)
    set_tests_properties(cubedsphere_mi_earth_tile_radial PROPERTIES
        PROCESSORS 12 TIMEOUT 180 ENVIRONMENT GAMERA_CS_TILE_TEST_MI=1)
    add_test(NAME cubedsphere_mi_earth_tile_angular
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 24
            $<TARGET_FILE:test_cubedsphere_mpi_earth_tile>
            1 2 2 72 16 16)
    set_tests_properties(cubedsphere_mi_earth_tile_angular PROPERTIES
        PROCESSORS 24 TIMEOUT 180 ENVIRONMENT GAMERA_CS_TILE_TEST_MI=1)
    add_test(NAME cubedsphere_mpi_tile_radial_64x24
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mpi_tile> 2 1 1 64 24 24)
    set_tests_properties(cubedsphere_mpi_tile_radial_64x24 PROPERTIES
        PROCESSORS 12 TIMEOUT 120)
    add_test(NAME cubedsphere_mpi_tile_angular_64x24
        COMMAND ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
            ${MPIEXEC_NUMPROC_FLAG} 12
            $<TARGET_FILE:test_cubedsphere_mpi_tile> 1 2 1 64 24 24)
    set_tests_properties(cubedsphere_mpi_tile_angular_64x24 PROPERTIES
        PROCESSORS 12 TIMEOUT 120)
    if(Python3_Interpreter_FOUND)
        add_test(NAME cubedsphere_earth_mi_tiled_physical_parity
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_tiled_mi_earth_parity.py
                $<TARGET_FILE:test_cubedsphere_dipole_background>
                $<TARGET_FILE:test_cubedsphere_mpi_earth_tile>
                ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
                ${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(cubedsphere_earth_mi_tiled_physical_parity
            PROPERTIES PROCESSORS 12 TIMEOUT 300)
        add_test(NAME cubedsphere_earth_mi_tiled_cadence_parity
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_tiled_mi_earth_parity.py
                $<TARGET_FILE:test_cubedsphere_dipole_background>
                $<TARGET_FILE:test_cubedsphere_mpi_earth_tile>
                ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
                ${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(cubedsphere_earth_mi_tiled_cadence_parity
            PROPERTIES PROCESSORS 12 TIMEOUT 600
            ENVIRONMENT GAMERA_CS_PARITY_STEPS=64)
        add_test(NAME cubedsphere_earth_mi_mpi_parity
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_mpi_earth_parity.py
                $<TARGET_FILE:test_cubedsphere_dipole_background>
                $<TARGET_FILE:test_cubedsphere_dipole_background_mpi>
                ${MPIEXEC_EXECUTABLE} ${CS_MPIEXEC_EXTRA_FLAGS}
                ${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(cubedsphere_earth_mi_mpi_parity PROPERTIES
            PROCESSORS 6 TIMEOUT 240)
    endif()
endif()
