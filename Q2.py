import time
from wordfreq import word_frequency
from PIL import Image


# Chapter 1: Q2

current_time = int(time.time())
generated_number = (current_time % 100) + 50

if generated_number % 2 == 0:
    generated_number += 10
print(generated_number)


def change_pixels_image() -> None:
    red_pixels = []
    """
    This function will change the r,g,b pixels of an image.
    """

    og_image = Image.open("chapter1.jpg")

    # add generated number to the r, g, b pixels of the image and create a new image
    image = og_image.copy()
    for i in range(image.width):
        for j in range(image.height):
            r, g, b = image.getpixel((i, j))
            r += generated_number
            g += generated_number
            b += generated_number
            red_pixels.append(r)
            image.putpixel((i, j), (r, g, b))

    image.save("chapter1out.png")
    print(f"sum of r pixel: {sum(red_pixels)}")


change_pixels_image()

# Chapter 2: Q2


def seprate_long_string(text: str) -> [str, str]:
    """
    This function will take a string and return two strings, one containing all the numbers and the other containing all the letters.
    :param text: str
    :return: [str, str]
    """
    numbers = []
    letters = []
    for i in text:
        if i.isdigit():
            numbers.append(i)
        else:
            letters.append(i)
    return ("".join(numbers), "".join(letters))


def convert_to_ascii(letters: str, numbers: str) -> str:
    """
    This function will take two strings, one containing numbers and the other containing
    letters and return a string containing the ascii values of the letters and numbers.
    :param letters: str
    :param numbers: str
    :return: tuple of lists

    """
    letters = [ord(letter) for letter in letters if letter.isupper()]
    numbers = [ord(number) for number in numbers if int(number) % 2 == 0]
    return letters, numbers


word = "56aAww1984sktr235270aYmn145ss785fsq31D0"

numbers, letters = seprate_long_string(word)
print(convert_to_ascii(letters, numbers))


# Program to showcase the required output of the ecrypted newspaper code


newspaper_code = """
VZ FRYSVFU VZCNGVRAG NAQ N YVGGYR VAFRPHER V ZNXR ZVFGNXRF V NZ BHG BS PBAGEBY
NAQNG GVZRF UNEQ GB UNAQYR OHG VS LBH PNAG UNAQYR ZR NG ZL JBEFG GURA LBH FHER NF
URYYQBAG QRFRER ZR NG ZL ORFG ZNEVYLA ZBAEBR
"""


def decrypt(text: str, shift: int) -> str:
    """
    This function will take a string and an integer and return the decrypted string.
    This function was from Q3.py of the assignment as it is the same as the decrypt function in Q3.py
    :param text: str
    :param shift: int
    :return: str
    """
    decrypted_text = ""
    for i in text:
        if i.isalpha():
            ascii_value = ord(i)
            ascii_value -= shift
            if i.islower():
                if ascii_value < ord("a"):
                    ascii_value += 26
            else:
                if ascii_value < ord("A"):
                    ascii_value += 26
            decrypted_text += chr(ascii_value)
        else:
            decrypted_text += i
    return decrypted_text


# determine the shift key


def shift_key(text: str) -> int:
    """
    This function will take a string and return the shift key.
    using the dictionary word frequency from the wordfreq module, the function will check if the first two words
    are in the english dictionary.
    :param text: str
    :return: int
    """
    shift = 0
    for i in range(1, 26):
        decrypted_text = decrypt(text, i)
        decrypted_text = decrypted_text.split()
        if word_frequency(decrypted_text[0], "en") and word_frequency(
            decrypted_text[1], "en"
        ):
            shift = i
            break
    return shift


print(shift_key(newspaper_code))
print(decrypt(newspaper_code, shift_key(newspaper_code)))

"""
OUTPUT:

IM SELFISH IMPATIENT AND A LITTLE INSECURE I MAKE MISTAKES I AM OUT OF CONTROL
ANDAT TIMES HARD TO HANDLE BUT IF YOU CANT HANDLE ME AT MY WORST THEN YOU SURE AS
HELLDONT DESERE ME AT MY BEST MARILYN MONROE

"""
