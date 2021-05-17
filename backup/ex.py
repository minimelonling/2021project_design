class e:
    def __init__(self, a, b):
        self.a = a
        self.b = b


a1 = e(1, None)
a2 = e(2, a1)
k = a2

while not k == None:
    print(k.a)
    k = k.b

def func(a):
    a.pop(0)

a = [1, 2, 3]
print(a)
a.pop(2)
print(a)
