# ---
# Initial 

book_num = 17866
book_link = f"https://www.gutenberg.org/ebooks/{book_num}.txt.utf-8"

# ---
# Proper nouns cleaning

persons_real_names = [
    "Charles II","Gladys Fleming","Lane Fleming","Jeff Rand","Arnold Rivers","Carl Gwinnett","Stephen Gresham","Philip Cabot","Nelda Dunmore","Fred Dunmore","Geraldine Varcek","Anton Varcek","Humphrey Goode","Dave Ritter","Mick McKenna","Carter Tipton","Irene Gresham","Dorothy Gresham","Adam Trehearne","Colin MacBride","Pierre Jarrett","Elmer Umholtz","Cecil Gillis","Karen Lawrence","Hester Prynne","Rudolf Hess","Jason Kirchner","Nell Gwyn","Gus Olsen","Jameson","Skinner","Pollard","Dunne","Harry Bentz","Davies","Kathie","Reuben"
]

name_to_replace = {
    # To be removed
    "Rand Dunmore": None, "Rand Pierre": None,"Mill Pack": None,"Fred Hines": None,"Sawyer": None,
    # To be replaced
    "Gladys": "Gladys Fleming","Mrs Fleming": "Gladys Fleming","Lane": "Lane Fleming","Mr Fleming": "Lane Fleming","Mick": "Mick McKenna","McKenna": "Mick McKenna","Jeff": "Jeff Rand","Rand": "Jeff Rand","Jefferson Davis": "Jeff Rand", "Rivers": "Arnold Rivers","Gwinnett": "Carl Gwinnett","Gresham": "Stephen Gresham","Philip": "Philip Cabot","Cabot": "Philip Cabot","Fred": "Fred Dunmore","Frederick Parker Dunmore": "Fred Dunmore","Geraldine": "Geraldine Varcek","Mrs Varcek": "Geraldine Varcek","Varcek": "Anton Varcek","Anton": "Anton Varcek","Humphrey": "Humphrey Goode","Goode": "Humphrey Goode","David Abercrombie Ritter": "Dave Ritter","Dave": "Dave Ritter","Tipton": "Carter Tipton","Trehearne": "Adam Trehearne","Colin": "Colin MacBride","MacBride": "Colin MacBride","Jarrett": "Pierre Jarrett","Umholtz": "Elmer Umholtz","Cecil": "Cecil Gillis","Gillis": "Cecil Gillis","Lawrence": "Karen Lawrence","Karen": "Karen Lawrence","Hester": "Hester Prynne","Aarvo": "Aarvo Kavaalen","Kavaalen": "Aarvo Kavaalen","Kirchner": "Jason Kirchner","Olsen": "Gus Olsen"    
}

# ---
# K-Core decomposition

class Point:
    def __init__(self, name, neighbors, pruned = False):
        self.name = name
        self.neighbors = neighbors
        self.degree = len(neighbors)
        self.pruned = pruned


