import num2words as n2w
from tkinter import *

# Function to convert number to words
# Function to convert number to words
# Function to convert number to words
def num_to_words():
    try:
        given_num = float(num.get())
        # Check if the number is non-positive or too large
        if given_num <= 0 or given_num > 1000000:
            raise ValueError("Number must be a positive number less than or equal to 1,000,000.")

        num_in_word = n2w.num2words(given_num)
        # Add newline after each line of output
        num_in_word_with_newlines = num_in_word.replace(' and ', '\n').replace(', ', '\n')
        display.config(text=str(num_in_word_with_newlines).capitalize())
    except ValueError as ve:
        # Check if the input contains non-numeric characters
        if not num.get().replace('.', '', 1).isdigit():
            display.config(text="Error: Please enter a valid number.")
        else:
            display.config(text="Error: " + str(ve))
    except OverflowError:
        display.config(text="Error: Number is too large. Please enter a number less than or equal to 1,000,000.")
    except Exception as e:
        display.config(text="Error: An unexpected error occurred.")


# Function to quit the application
def quit_application():
    root.destroy()

# Create Tkinter window
root = Tk()
root.title("Numbers to Words")
root.geometry("650x400")

# StringVar to store input number
num = StringVar()

# Adding title
title = Label(root, text="Number to Words converter",
               fg="Blue", font=("Arial", 20, 'bold'))
title.pack(pady=10)

# Options
formats_label = Label(root, text="Formats supported :  ",
           fg="green", font=("Arial", 10, 'bold'))
formats_label.pack()

# Display supported formats
supported_formats = ["1. Positives", "2. Negatives", "3. Zeros", "4. Floating points/decimals/fractions"]
for format_text in supported_formats:
    Label(root, text=format_text, fg="green", font=("Arial", 10, 'bold')).pack()

# Label for input number
num_entry_label = Label(root, text="Enter a number :", fg="Blue", font=("Arial", 15, 'bold'))
num_entry_label.pack(pady=10)

# Entry widget for user input
num_entry = Entry(root, textvariable=num, width=30)
num_entry.pack()

# Button to trigger conversion
btn = Button(master=root, text="Calculate", fg="green", font=("Arial", 10, 'bold'), command=num_to_words)
btn.pack(pady=10)

# Display label
display = Label(root, text="", fg="black", font=("Arial", 10, 'bold'))
display.pack(fill=BOTH, expand=True)

# Quit button
quit_btn = Button(root, text="Quit", fg="red", font=("Arial", 10, 'bold'), command=quit_application)
quit_btn.pack(side=BOTTOM, pady=10)

# Set icon photo
try:
    photo = PhotoImage(file="C:/Coding/Python Projects/NumberToWords/number.png")
    root.iconphoto(False, photo)
except Exception as e:
    print("Error in setting icon photo:", e)

# Start the GUI event loop
root.mainloop()
