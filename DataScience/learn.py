filename = input("Enter the file name\n")
try:
    fhandle = open(filename,"r")
except:
    print("The file couldn't be opened")
    quit()
count = {}
for line in fhandle:
    l = line.split()
    for word in l:
        count[word]=count.get(word,0) + 1
list2 = sorted([(v, k) for k, v in count.items()], reverse=True)
for items in range(5):
    print(list2[items])








