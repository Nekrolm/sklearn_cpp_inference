#pragma once
#ifdef _WIN32
#    ifdef SKLEARN_EXPORTS
#        define SKLEARN_API __declspec(dllexport)
#    else
#        define SKLEARN_API __declspec(dllimport)
#    endif
#else
#    define SKLEARN_API
#endif
