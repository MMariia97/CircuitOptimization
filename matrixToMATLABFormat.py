import numpy as np

def main():
    str1 = ""
    outerList = [[1, 1, 1], [4, 3, 2]]
    for innerListId in range(len(outerList)):
        for elId in range(len(outerList[innerListId])):
            str1+=str(outerList[innerListId][elId])
            if (elId < (len(outerList[innerListId])-1)):
                str1+=","
        if (innerListId < (len(outerList)-1)):
            str1+=";"
    print(str1)

if __name__ == '__main__':
    main()
