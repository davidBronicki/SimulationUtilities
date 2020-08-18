
class polynomial:
	def __init__(self, initPoly):
		self.coefs = initPoly

	def __add__(self, other):
		pad = abs(len(self.coefs) - len(other.coefs)) * [0]
		output = []
		for item in zip(self.coefs + pad, other.coefs + pad):
			output.append(item[0] + item[1])
		return polynomial(output)

	def __mul__(self, other):
		output = polynomial([0])
		for i in range(len(other.coefs)):
			item = self.coefs.copy()
			for j in range(len(item)):
				item[j] *= other.coefs[i]
			output += polynomial(i * [0] + item)
		return output

	def __str__(self):
		return str(self.coefs)


poly1 = polynomial(6 * [1])

polyOut = polynomial([1])

for i in range(3):
	polyOut *= poly1

print(polyOut)