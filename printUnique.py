input_file = open("input/input_Dict.csv", 'r')
file_contents = input_file.read()
input_file.close()
duplicates = []
word_list = file_contents.split()
file = open("unique.txt", 'w')
for word in word_list:
    if word not in duplicates:
        duplicates.append(word)
        file.write(str(word) + "\n")
file.close()