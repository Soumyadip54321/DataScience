
class PartyAnimal:
    x = 0
    name = ""
    def __init__(self,Name):
        self.name = Name
        print("I am constructed")

    def party(self):
        self.x += 5
        print(self.x)

    def __del__(self):
        print("I am destructed and my details were",self.name," and",self.x)


class NewParty(PartyAnimal):
    points = 0

    def completelynew(self):
        self.points += 10
        self.party()
        print(self.x, self.points, self.name)


an = NewParty("Diganta")
an.completelynew()


