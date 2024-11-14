set(HIGHS_EXTRA_UNIT_TESTS
    ${HIGHS_SOURCE_DIR}/check/highs-unit-tests/TestCppExtraDummy.cpp
    ${HIGHS_SOURCE_DIR}/check/highs-unit-tests/TestCppExtraGpuOnly.cpp
)


if (BUILD_CXX)
    if (CUPDLP_GPU)
        set(HIGHS_EXTRA_UNIT_TESTS
            ${HIGHS_SOURCE_DIR}/check/highs-unit-tests/TestGpuFireUp.cpp
        )
    endif()

    set(HIGHS_EXTRA_UNIT_TESTS
        ${HIGHS_EXTRA_UNIT_TESTS} 
        ${HIGHS_SOURCE_DIR}/check/highs-unit-tests/TestExtra.cpp
    )
endif()

message(STATUS "DEFINE TEST extra: ${HIGHS_EXTRA_UNIT_TESTS}")
