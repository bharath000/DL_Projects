import pandas as pd 

# labels = pd.read_csv("D:/MSCS/ADL/Data/train.csv")
# print(len(labels))
# print(labels.iloc[[0,1], 0])
# print(labels.iloc[[0,1], 1])



# f = 3, cit = [1,3,6,4]

# n = len(1,3,6,4)
# res =[]
# def feul(s,d):
#     return s-d

# def backtrack(source,destinations,re,f,j):

#     #cost = fuel(source,destinations[0])
#     if destinations:
#         cost = fuel(source,destinations[j])
#         if f == cost:
#             re.append(cit[i+1])
#             f = 0
#         if f < cost:
#             pass #check for other case
#             backtrack(source,destination,re,f)

#         if f > cost:
#             pass # continue for going forward and also update f
#             re.append(destinations[j])
#             backtrack(destinations[j],destinations[j+1:],re,f-c)
#     else:
#         return

    
    


# for i in range(0,n-1):
#     j = 0
#     backtrack(cit[0], cit[i+1:], res, f,j)





# def slowestKey(keyTimes):
#     alphabet = 'abcdefghijklmnopqrstuvwxyz'

#     alpha_dict = {}
#     for i in range(0,len(alphabet)):
#         alpha_dict[i] = alphabet[i]
    




#     for i in range(len(keyTimes)-1, 0, -1):
#         keyTimes[i][1] -= keyTimes[i-1][1]
#     return alpha_dict[max(keyTimes, key = lambda x : x[1])[0]]


# print(slowestKey([[0,2],[1,3],[0,7]]))
# print(slowestKey([[0,1],[0,3],[4,5],[5,6],[4,10]]))


# dict1 = {}
# for i in products:

#     if i in dict1.keys():
#         dict1[i] += 1
#     else:
#         dict1[i] = 1
for i in range(0,1):
    print(i)

exit()


from functools import cmp_to_key

def compare(item1, item2):
    if item1[1] != item2[1]:
        return item1[1] - item2[1]
    else:
        if item1[0] <= item2[0]:
            return -1
        else:
            return 1

def featuredProducts(products):

    freq = {}

    for item in products:
        if item not in freq:
            freq[item] = 0
        freq[item] += 1

    list_p = []

    for key,value in freq.items():
        list_p.append((key, value))

    list_p = sorted(list_p, key=cmp_to_key(compare))

    return list_p[-1][0]


lis = ["greenShirt", "bluePants","redShirt","blackShoes","redPants","redPants","whiteShirt","redShirt"]


print(featuredProducts(lis))

# def slowestKey(keyTimes):
#
#     alphabet = 'abcdefghijklmnopqrstuvwxyz'
#
#     alpha_dict = {}
#     for i in range(0,len(alphabet)):
#         alpha_dict[i] = alphabet[i]
#
#     for i in range(len(keyTimes)-1, 0, -1):
#         keyTimes[i][1] -= keyTimes[i-1][1]
#
#     return alpha_dict[max(keyTimes, key = lambda x : x[1])[0]]
#
#
# print(slowestKey([[0,2],[1,3],[0,7]]))
# print(slowestKey([[0,1],[0,3],[4,5],[5,6],[4,10]]))



