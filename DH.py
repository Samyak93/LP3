from __future__ import print_function
# Variables Used
sharedPrime = 23
sharedBase = 5
aliceSecret = 6
bobSecret = 15
# Begin
print( "Publicly Shared Variables:")
print( "Publicly Shared Prime: " , sharedPrime )
print( "Publicly Shared Base: " , sharedBase )

# Alice Sends Bob A = g^a mod p
A = (sharedBase**aliceSecret) % sharedPrime
print  ("\n Alice Sends Over Public Chanel: " , A )

# Bob Sends Alice B = g^b mod p
B = (sharedBase ** bobSecret) % sharedPrime
print(" \n Bob Sends Over Public Chanel: ", B )

print( "\n------------\n" )
print( "Privately Calculated Shared Secret:" )
# Alice Computes Shared Secret: s = B^a mod p
aliceSharedSecret = (B ** aliceSecret) % sharedPrime
print( "Alice Shared Secret: ", aliceSharedSecret )

bobSharedSecret = (A**bobSecret) % sharedPrime
print( " Bob Shared Secret: ", bobSharedSecret )