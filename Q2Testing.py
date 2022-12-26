
def Rome2Now(number):
    i = 0
    value = 0
    while i < (len(number)):
        if number[i] == "I":
            if number[i+1] == "V":
                value += 4
                i += 1
            elif number[i+1] == "X":
                value += 9
                i += 1
            else:
                value += 1        
        elif number[i] == "V":
            value += 5
        elif number[i] == "X":
            if (number[i+1] == "L" and (i+2) > len(number)):
                value += 40
                i += 1
            elif (number[i+1] == "C" and (i+2) > len(number)):
                value += 90
                i += 1
            else:
                value += 10
        elif number[i] == "L":
            value += 50
        elif number[i] == "C":
            if (number[i+1] == "D" and (i+2) > len(number)):
                value += 400
                i += 1
            elif (number[i] == "M" and (i+2) > len(number)):
                value += 900
                i += 1
            else:
                value += 100
        elif number[i] == "D":
            value += 500
        elif number[i] == "M":
            value += 1000
        else:
            print("Error input")
            break
        i += 1
    return value

number = "XXCM"
print(Rome2Now(number))
