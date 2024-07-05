#include <stdio.h>
#include <stdbool.h>

void compare_arrays(int* arr1, int* arr2, int size)
{
    bool match = true;
    for (int i = 0; i < size; i++)
    {
        if (arr1[i] != arr2[i])
        {
            match = false;
            break;
        }
    }

    if (match)
    {
        printf("Arrays match.\n");
    }
    else
    {
        printf("Arrays do not match.\n");
    }
}
