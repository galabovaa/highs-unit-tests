
set(HIGHS_EXTRA_UNIT_TESTS
    ${HIGHS_SOURCE_DIR}/check/highs-unit-tests/TestGpuFireUp.cpp
    )

if (BUILD_CXX)
    set(HIGHS_EXTRA_UNIT_TESTS
        ${HIGHS_EXTRA_UNIT_TESTS} 
        ${HIGHS_SOURCE_DIR}/check/highs-unit-tests/TestExtra.cpp
    )
endif()

message(STATUS "DEFINE TEST extra: ${HIGHS_EXTRA_UNIT_TESTS}")
