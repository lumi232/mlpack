/**
 * @param mlpack_cli_main.hpp
 * @author Ryan Curtin
 *
 * This file, based on the value of the macro BINDING_TYPE, will define the
 * macros necessary to compile an mlpack binding for the target language.
 *
 * This file should *only* be included by a program that is meant to be a
 * command-line program or a binding to another language.  This file also
 * includes param_checks.hpp, which contains functions that are used to check
 * parameter values at runtime.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_MLPACK_MAIN_HPP
#define MLPACK_CORE_UTIL_MLPACK_MAIN_HPP

#define BINDING_TYPE_CLI 0
#define BINDING_TYPE_TEST 1
#define BINDING_TYPE_PYX 2
#define BINDING_TYPE_JL 3
#define BINDING_TYPE_MARKDOWN 128
#define BINDING_TYPE_UNKNOWN -1

#ifndef BINDING_TYPE
#define BINDING_TYPE BINDING_TYPE_UNKNOWN
#endif

#if (BINDING_TYPE == BINDING_TYPE_CLI) // This is a command-line executable.

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/cli/cli_option.hpp>
#include <mlpack/bindings/cli/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::cli::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::cli::PrintValue
#define PRINT_CALL mlpack::bindings::cli::ProgramCall
#define PRINT_DATASET mlpack::bindings::cli::PrintDataset
#define PRINT_MODEL mlpack::bindings::cli::PrintModel
#define BINDING_IGNORE_CHECK mlpack::bindings::cli::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::cli::CLIOption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>
#include <mlpack/bindings/cli/parse_command_line.hpp>
#include <mlpack/bindings/cli/end_program.hpp>

static void mlpackMain(); // This is typically defined after this include.

int main(int argc, char** argv)
{
  // Parse the command-line options; put them into CLI.
  mlpack::bindings::cli::ParseCommandLine(argc, argv);
  // Enable timing.
  mlpack::Timer::EnableTiming();

  // A "total_time" timer is run by default for each mlpack program.
  mlpack::Timer::Start("total_time");

  mlpackMain();

  // Print output options, print verbose information, save model parameters,
  // clean up, and so forth.
  mlpack::bindings::cli::EndProgram();
}

#elif(BINDING_TYPE == BINDING_TYPE_TEST) // This is a unit test.

// Matrices are not transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED false

#include <mlpack/bindings/tests/test_option.hpp>
#include <mlpack/bindings/tests/ignore_check.hpp>
#include <mlpack/bindings/tests/clean_memory.hpp>

// These functions will do nothing.
#define PRINT_PARAM_STRING(A) std::string(" ")
#define PRINT_PARAM_VALUE(A, B) std::string(" ")
#define PRINT_DATASET(A) std::string(" ")
#define PRINT_MODEL(A) std::string(" ")
#define PRINT_CALL(...) std::string(" ")
#define BINDING_IGNORE_CHECK mlpack::bindings::tests::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::tests::TestOption<T>;

}
}

// testName symbol should be defined in each binding test file
#include <mlpack/core/util/param.hpp>

#undef PROGRAM_INFO
#define PROGRAM_INFO(NAME, SHORT_DESC, DESC, ...) \
    static mlpack::util::ProgramDoc \
    cli_programdoc_dummy_object = mlpack::util::ProgramDoc(NAME, SHORT_DESC, \
    []() { return DESC; }, { __VA_ARGS__ })

#elif(BINDING_TYPE == BINDING_TYPE_PYX) // This is a Python binding.

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/python/py_option.hpp>
#include <mlpack/bindings/python/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::python::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::python::PrintValue
#define PRINT_DATASET mlpack::bindings::python::PrintDataset
#define PRINT_MODEL mlpack::bindings::python::PrintModel
#define PRINT_CALL mlpack::bindings::python::ProgramCall
#define BINDING_IGNORE_CHECK mlpack::bindings::python::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::python::PyOption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>

#undef PROGRAM_INFO
#define PROGRAM_INFO(NAME, SHORT_DESC, DESC, ...) \
    static mlpack::util::ProgramDoc \
    cli_programdoc_dummy_object = mlpack::util::ProgramDoc(NAME, SHORT_DESC, \
    []() { return DESC; }, { __VA_ARGS__ }); \
    namespace mlpack { \
    namespace bindings { \
    namespace python { \
    std::string programName = NAME; \
    } \
    } \
    }

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");
PARAM_FLAG("copy_all_inputs", "If specified, all input parameters will be deep"
    " copied before the method is run.  This is useful for debugging problems "
    "where the input parameters are being modified by the algorithm, but can "
    "slow down the code.", "");

// Nothing else needs to be defined---the binding will use mlpackMain() as-is.

#elif(BINDING_TYPE == BINDING_TYPE_JL) // This is a Julia binding.

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/julia/julia_option.hpp>
#include <mlpack/bindings/julia/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::julia::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::julia::PrintValue
#define PRINT_DATASET mlpack::bindings::julia::PrintDataset
#define PRINT_MODEL mlpack::bindings::julia::PrintModel
#define PRINT_CALL mlpack::bindings::julia::ProgramCall
#define BINDING_IGNORE_CHECK mlpack::bindings::julia::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::julia::JuliaOption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>

#undef PROGRAM_INFO
#define PROGRAM_INFO(NAME, SHORT_DESC, DESC, ...) static \
    mlpack::util::ProgramDoc \
    cli_programdoc_dummy_object = mlpack::util::ProgramDoc(NAME, SHORT_DESC, \
    []() { return DESC; }, { __VA_ARGS__ }); \
    namespace mlpack { \
    namespace bindings { \
    namespace julia { \
    std::string programName = NAME; \
    } \
    } \
    }

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

#elif BINDING_TYPE == BINDING_TYPE_MARKDOWN

// We use BINDING_NAME in PROGRAM_INFO() so it needs to be defined.
#ifndef BINDING_NAME
  #error "BINDING_NAME must be defined when BINDING_TYPE is Markdown!"
#endif

// This value doesn't actually matter, but it needs to be defined as something.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/markdown/md_option.hpp>
#include <mlpack/bindings/markdown/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::markdown::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::markdown::PrintValue
#define PRINT_DATASET mlpack::bindings::markdown::PrintDataset
#define PRINT_MODEL mlpack::bindings::markdown::PrintModel
#define PRINT_CALL mlpack::bindings::markdown::ProgramCall
#define BINDING_IGNORE_CHECK mlpack::bindings::markdown::IgnoreCheck

// This doesn't actually matter for this binding type.
#define BINDING_MATRIX_TRANSPOSED true

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::markdown::MDOption<T>;

}
}

#include <mlpack/core/util/param.hpp>
#include <mlpack/bindings/markdown/program_doc_wrapper.hpp>

#undef PROGRAM_INFO
#define PROGRAM_INFO(NAME, SHORT_DESC, DESC, ...) static \
    mlpack::bindings::markdown::ProgramDocWrapper \
    cli_programdoc_dummy_object = \
    mlpack::bindings::markdown::ProgramDocWrapper(BINDING_NAME, NAME, \
    SHORT_DESC, []() { return DESC; }, { __VA_ARGS__ }); \

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

// CLI-specific parameters.
PARAM_FLAG("help", "Default help info.", "h");
PARAM_STRING_IN("info", "Print help on a specific option.", "", "");
PARAM_FLAG("version", "Display the version of mlpack.", "V");

// Python-specific parameters.
PARAM_FLAG("copy_all_inputs", "If specified, all input parameters will be deep"
    " copied before the method is run.  This is useful for debugging problems "
    "where the input parameters are being modified by the algorithm, but can "
    "slow down the code.", "");

#else

#error "Unknown binding type!  Be sure BINDING_TYPE is defined if you are " \
       "including <mlpack/core/util/mlpack_main.hpp>.";

#endif

#include "param_checks.hpp"

#endif
