def encrypt(text, key):
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            shifted = ord(char) + key
            if char.islower():
                if shifted > ord("z"):
                    shifted -= 26
                elif shifted < ord("a"):
                    shifted += 26
            else:
                if shifted > ord("Z"):
                    shifted -= 26
                elif shifted < ord("A"):
                    shifted += 26
            encrypted_text += chr(shifted)
        else:
            encrypted_text += char
    return encrypted_text


# The decrypt function is similar to the encrypt function, but it subtracts the key value from the ASCII value of each character instead of adding it.
def decrypt(text, key):
    decrypted_text = ""
    for char in text:
        if char.isalpha():
            shifted = ord(char) - key
            if char.islower():
                if shifted > ord("z"):
                    shifted -= 26
                elif shifted < ord("a"):
                    shifted += 26
            else:
                if shifted > ord("Z"):
                    shifted -= 26
                elif shifted < ord("A"):
                    shifted += 26
            decrypted_text += chr(shifted)
        else:
            decrypted_text += char
    return decrypted_text


encrypted_code = """

tybony_inevnoyr = 100
zl_qvpg = {
    'xrl1': 'inyhr1',
    'xrl2': 'inyhr2',
    'xrl3': 'inyhr3'
}

qrs cebprff_ahzoref():
    tybony tybony_inevnoyr
    ybpny_inevnoyr = 5
    ahzoref = [1, 2, 3, 4, 5]

    juvyr ybpny_inevnoyr > 0:
        vs ybpny_inevnoyr % 2 == 0:
            ahzoref.erzbir(ybpny_inevnoyr)
        ybpny_inevnoyr -= 1
    erghea ahzoref

zl_frg = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1}
erfhyg = cebprff_ahzoref(ahzoref=zl_frg)

qrs zbqvsl_qvpg():
    ybpny_inevnoyr = 10
    zl_qvpg['xrl4'] = ybpny_inevnoyr

zbqvsl_qvpg(5)

qrs hcqngr_tybony():
    tybony tybony_inevnoyr
    tybony_inevnoyr += 10

sbe v va enatr(5):
    cevag(v)
    v +=1

vs zl_frg vf abg Abar naq zl_qvpg['xrl4'] == 10:
    cevag("Pbaqvgvba zrg!")

vs 5 abg va zl_qvpg:
    cevag("5 abg sbhaq va gur qvpgvbanel!")

cevag(tybony_inevnoyr)
cevag(zl_qvpg)
cevag(zl_frg)

"""


total = 0

for i in range(5):
    for j in range(3):
        if i + j == 5:
            total += i + j
        else:
            total -= i - j

counter = 0

while counter < 5:
    if total < 13:
        total += 1
    elif total > 13:
        total -= 1
    else:
        counter += 2

print(decrypt(encrypted_code, total))


# Output:

global_variable = 100
my_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}


# The process function was missing the numbers parameter
def process_numbers(numbers):
    global global_variable
    local_variable = 5
    # The numbers variable is alread passed as a parameter, delcaring it here will
    # overwrite the passed parameter, so i recommend removing this line
    # numbers = [1, 2, 3, 4, 5]

    while local_variable > 0:
        if local_variable % 2 == 0:
            numbers.remove(local_variable)
        local_variable -= 1
    return numbers


my_set = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1}
result = process_numbers(numbers=my_set)


def modify_dict():
    local_variable = 10
    my_dict["key4"] = local_variable


# The function was called with a parameter, but the function does not accept any parameters,
# so i recommend removing the parameter
modify_dict()


def update_global():
    global global_variable
    global_variable += 10


for i in range(5):
    print(i)
    i += 1

if my_set is not None and my_dict["key4"] == 10:
    print("Condition met!")

# The condition should check if 5 is not in the values of the dictionary
if 5 not in my_dict.values():
    print("5 not found in the dictionary!")

print(global_variable)
print(my_dict)
print(my_set)
