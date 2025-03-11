def bubble_sort(arr):
    # Get the length of the list
    length = len(arr)
    # Traverse through all elements in the list
    for i in range(length):
        # Last i elements are already in place, so we don't need to check them
        for j in range(0, length - i - 1):
            # Traverse the list from 0 to length-i-1
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Example usage:
my_list = [5, 2, 8, 3, 1]
print("Original list:", my_list)
bubble_sort(my_list)
print("Sorted list:", my_list)
