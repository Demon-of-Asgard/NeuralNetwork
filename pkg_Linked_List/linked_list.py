#-------------------------------------------------------------------------------
class LinkedList():
    def __init__(self):
         self.links = []

    def add_link(self, item, loc:int = -1):
        self.links.insert(loc, item)

    def remove_link(self, loc:int = -1):
        self.links.pop(loc)

    def get_link(self, loc:int = 0):
        return self.links[loc]

#-------------------------------------------------------------------------------
