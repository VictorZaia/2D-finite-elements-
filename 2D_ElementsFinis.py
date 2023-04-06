
import numpy as np
import matplotlib.pyplot as plt

"""
PROBLEM DATA:
"""

"Problem dimensions and size of mesh"
d1 = 1 #m, horizontal dimension
d2 = 1 #m, vertical dimension
p = 5 #horizontal
m = 5 #vertical

"Material data"
E =  2E11 #Young modulus [Pa]
mu = 0.3 #poisson ratio
t = 0.01 #thickness [m]

"Gauss points per element"
GPE = 4 #Gauss points per element

"Force vector [node id, force in x direction, force in y direction]"
force = [] #[N]
force.append([6, -10000, 10000])
force.append([12, -10000, 10000])
force.append([18, -10000, 10000])
force.append([24, -10000, 10000])
force = np.array(force)



"""
DISCRETIZATION OF THE PROBLEM:
"""

"discretization of the problem"
def maillage(d1,d2,p,m):
    "Problem dimension"
    PD = 2

    "Number of nodes, number of elements and number of nodes per element"
    NoN = (p+1)*(m+1)
    NoE = p*m 
    NPE = 4

    "creation of the nodes"
    NL = np.zeros([NoN,PD])  
    
    "Size of the element"
    a = d1/p
    b = d2/m

    n = 0 #number of the line of the list of nodes
    
    for i in range(0,m+1):
        for j in range(0,p+1):
            NL[n,0] = (j)*a
            NL[n,1] = (i)*b
            n+=1
    
    "creation of the elements"
    EL = np.zeros([NoE,NPE])
    
    for i in range(1,m+1):
        for j in range(1,p+1):
            if j == 1:
                EL[(i-1)*p,0] = (i-1)*(p+1) + j
                EL[(i-1)*p,1] = EL[(i-1)*p+j-1,0] + 1
                EL[(i-1)*p,3] = EL[(i-1)*p+j-1,0] + (p+1)
                EL[(i-1)*p,2] = EL[(i-1)*p+j-1,3] + 1
            else:
                EL[(i-1)*p+j-1,0] = EL[(i-1)*p+j-2,1]
                EL[(i-1)*p+j-1,3] = EL[(i-1)*p+j-2,2]
                EL[(i-1)*p+j-1,1] = EL[(i-1)*p+j-1,0] + 1
                EL[(i-1)*p+j-1,2] = EL[(i-1)*p+j-1,3] + 1     
    
    return(NL,EL)

def plotMesh(EL, NL, color, numb):
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim(0, np.max(NL))
    plt.ylim(0, np.max(NL))
    
    if numb == True:
        "Ploting the node number"
        count = 1
        for i in range(0, np.size(NL, 0)):
            ax.annotate(count, xy = (NL[i,0], NL[i,1]))
            count += 1
    
        "Ploting the elements number"
        count = 1
        for j in range(0, np.size(EL, 0)):
            plt.annotate(count, xy = ((NL[int(EL[j,0])-1,0] + NL[int(EL[j,1])-1,0] + NL[int(EL[j,2])-1,0] + NL[int(EL[j,3])-1,0])/4, (NL[int(EL[j,0])-1,1] + NL[int(EL[j,1])-1,1] + NL[int(EL[j,2])-1,1] + NL[int(EL[j,3])-1,1])/4))
            count += 1

    
    "ploting the lines"
    x0, y0 = NL[EL[:,0].astype(int) - 1, 0], NL[EL[:,0].astype(int) - 1,1]
    x1, y1 = NL[EL[:,1].astype(int) - 1, 0], NL[EL[:,1].astype(int) - 1,1]
    x2, y2 = NL[EL[:,2].astype(int) - 1, 0], NL[EL[:,2].astype(int) - 1,1]
    x3, y3 = NL[EL[:,3].astype(int) - 1, 0], NL[EL[:,3].astype(int) - 1,1]
    plt.plot(np.array([x0, x1]), np.array([y0, y1]), color, linewidth=3)
    plt.plot(np.array([x1, x2]), np.array([y1, y2]), color, linewidth=3)
    plt.plot(np.array([x2, x3]), np.array([y2, y3]), color, linewidth=3)
    plt.plot(np.array([x3, x0]), np.array([y3, y0]), color, linewidth=3)



"""
COMPUTING THE STIFFNESS OF A ELEMENT:
"""

"Function that compute the xi, eta and  for each integration point"
def defineIntegrationPoints(NPE, GPE, GP):
    
    if NPE == 4: #Q4
        if GPE == 1: #number of integration points on the element
            if GP == 1:
                xi = 0
                eta = 0
                alpha = 4
        if GPE == 4: #number of integration points on the element
            if GP == 1:
                xi = -1 / np.sqrt(3)
                eta = -1 / np.sqrt(3)
                alpha = 1
            if GP == 2:
                xi = 1 / np.sqrt(3)
                eta = -1 / np.sqrt(3)
                alpha = 1
            if GP == 3:
                xi = 1 / np.sqrt(3)
                eta = 1 / np.sqrt(3)
                alpha = 1
            if GP == 4:
                xi = -1 / np.sqrt(3)
                eta = 1 / np.sqrt(3)
                alpha = 1
    return (xi, eta, alpha)

"Function that computes the Shape functions on the natural coordinates"
def defineGrad_N_Nat(NPE, xi, eta):
    
    PD = 2 #Problem dimension
    N_Nat = np.zeros([PD, NPE]) #shape funcitons on the natural coordinate
    
    if NPE ==4:
        N_Nat[0,0] = -1/4 * (1 - eta)
        N_Nat[0,1] = 1/4 * (1 - eta)
        N_Nat[0,2] = 1/4 * (1 + eta)
        N_Nat[0,3] = -1/4 * (1 + eta)
        
        N_Nat[1,0] = -1/4 * (1 - xi)
        N_Nat[1,1] = -1/4 * (1 + xi)
        N_Nat[1,2] = 1/4 * (1 + xi)
        N_Nat[1,3] = 1/4 * (1 - xi)

    return(N_Nat)

"functions that computes the product of J-1 and N_Grad_Nat"
def defineDG(J, grad_N_nat):
    DG = np.matmul(np.linalg.inv(J).T, grad_N_nat)
    return(DG)

"function that computes the matrix B"
def defineB(DG):
    dN1x = DG[0,0]
    dN1y = DG[1,0]
    
    dN2x = DG[0,1]
    dN2y = DG[1,1]
    
    dN3x = DG[0,2]
    dN3y = DG[1,2]
    
    dN4x = DG[0,3]
    dN4y = DG[1,3]
    
    B = np.array([[dN1x, 0, dN2x, 0, dN3x, 0, dN4x,0], [0, dN1y, 0, dN2y, 0, dN3y, 0, dN4y], [dN1y, dN1x, dN2y, dN2x, dN3y, dN3x, dN4y, dN4x]])
    return(B)

"Function that computes the jacobian"
def defineJ(coord, grad_N_Nat):
    J = np.matmul(coord, grad_N_Nat.T)
    return(J)

"Function that computes the constitutive D - Plane stress"
def defineConstitutiveD(E, mu):
    D = np.zeros([3, 3])
    aux = E / (1 - mu**2)
    
    D[0,0] = 1
    D[1,0] = mu
    D[2,0] = 0
    
    D[0,1] = mu
    D[1,1] = 1
    D[2,1] = 0
    
    D[0,2] = 0
    D[1,2] = 0
    D[2,2] = (1 - mu) / 2
    
    D = aux * D
    return(D)

"function that computes the stifness matrix for 1 element"
def calculateGlobalStiffness(x, GPE):
    
    NPE = np.size(x,0) #nodes per element
    PD = np.size(x,1) # Problem dimension
    K = np.zeros([PD*NPE, PD*NPE])
    coord = x.T #coordinates of the nodes
    for GP in range(1, GPE+1):
        (xi, eta, alpha) = defineIntegrationPoints(NPE, GPE, GP)
        grad_N_Nat = defineGrad_N_Nat(NPE, xi, eta)
        J = defineJ(coord, grad_N_Nat)
        DG = defineDG(J, grad_N_Nat)
        B = defineB(DG)
        D = defineConstitutiveD(E, mu)
        
        aux = np.matmul(D, B)
        aux = np.matmul(B.T, aux)
        aux = np.linalg.det(J) * aux * alpha * t
        
        K = K + aux
    return(K)



"""
ASSEMBLY OF THE MODEL:
"""

"Assining the boundary conditions to the nodes on the left of the beam"    
def defineBoundaryConditions(NL, p, m):
    "Number of degrees of freedom per node, in 2d Q4: u and v"
    NDOF = 2
    DOF = np.zeros([len(NL), NDOF+1])

    for i in range(0, len(NL)):
        DOF[i,0] = i+1

    for n in range(0, m + 1):
        if n == 0:
            DOF[0,:] = 1
        else:
            DOF[((p+1) * n), 1] = 1
            DOF[((p+1) * n), 2] = 1
    return(DOF)

"Creating the equations that will be needed to compute the matrix kuu"
def getEquations(NL):
    numberOfReactions = 0
    numberOfDof = 0
    intlist = np.zeros([len(NL),2])

    DOF = defineBoundaryConditions(NL,p,m)
    for i in range(0,len(NL)):
        for j in range(1, np.size(NL, 1) + 1):
            if DOF[i,j] == 1:
                numberOfReactions -= 1
                intlist[i,j-1] = numberOfReactions;
            else:
                numberOfDof += 1
                intlist[i,j-1] = numberOfDof;

    return(intlist)

"Matrix kuu which is the stiffness matrix linked to the nodes with displacement diferent of 0"
def calculateKuu(EL, NL):
    equations = getEquations(NL)
    numberOfDof = np.max(equations)
    kuu = np.zeros([int(numberOfDof),int(numberOfDof)])
    x = np.zeros([np.size(EL,1),2])

    for i in range(0,np.size(EL,0)):
        nodes = EL[i]
        listeq = np.zeros([np.size(EL,1),2])
        
        "Loop to get the coordenates of each node that make the element and the equation."
        for j in range(0, len(nodes)):
            x[j] = NL[int(nodes[j]-1)]
        for k in range(0,len(nodes)):
            listeq[k] = equations[int(nodes[k]-1)]

        listeq = listeq.flatten()

        "Computing of the stiffness matrix for the element."
        K = calculateGlobalStiffness(x, 4)

        "Loop to verify the equations and add the stiffness to the global matrix."
        for m in range(0,np.size(K,0)):
            eqline = int(listeq[m])
            for n in range(0,np.size(K,1)):
                eqcol = int(listeq[n])
                if eqline > 0 and eqcol > 0:
                    aux = kuu[(eqline - 1), (eqcol - 1)]
                    aux += K[m, n]
                    kuu[(eqline - 1), (eqcol - 1)] =  aux
                   
    return(kuu)

"Computing the assembly of the global stiffness matrix"
def calculateK(EL, NL):
    equations = getEquations(NL)
    numberOfDof = np.max(equations)
    numberOfReactions = np.min(equations)
    kt = np.zeros([int(numberOfDof + (-1) * numberOfReactions ),int(numberOfDof + (-1)* numberOfReactions)])
    x = np.zeros([np.size(EL,1), 2])

    for i in range(0,np.size(EL,0)):
        nodes = EL[i]
        listeq = np.zeros([np.size(EL,1), 2])
        
        "Loop to get the coordenates of each node that make the element and the equation."
        for j in range(0, len(nodes)):
            x[j] = NL[int(nodes[j]-1)]
        for k in range(0,len(nodes)):
            listeq[k] = equations[int(nodes[k]-1)]

        listeq = listeq.flatten()

        "Computing of the stiffness matrix for the element."
        K = calculateGlobalStiffness(x, 4)

        "Loop to verify the equations and add the stiffness to the global matrix."
        for m in range(0,np.size(K,0)):
            eqline = int(listeq[m])
            for n in range(0,np.size(K,1)):
                eqcol = int(listeq[n])
                if eqline > 0 and eqcol > 0:
                    aux = kt[(eqline - 1), (eqcol - 1)]
                    aux += K[m, n]
                    kt[(eqline - 1), (eqcol - 1)] =  aux
                    
                elif eqline < 0 and eqcol > 0:
                   aux = kt[((-1) * eqline - 1 + int(numberOfDof)), (eqcol - 1)]
                   aux += K[m, n]
                   kt[((-1) * eqline - 1 + int(numberOfDof)), (eqcol - 1)] =  aux
                   
                elif eqline > 0 and eqcol < 0:
                    aux = kt[(eqline - 1), ((-1) * eqcol - 1 + int(numberOfDof))]
                    aux += K[m, n]
                    kt[(eqline - 1), ((-1) * eqcol - 1 + int(numberOfDof))] =  aux
                    
                elif eqline < 0 and eqcol < 0:
                    aux = kt[((-1) * eqline - 1 + int(numberOfDof)), ((-1) * eqcol - 1 + int(numberOfDof))]
                    aux += K[m, n]
                    kt[((-1) * eqline - 1 + int(numberOfDof)), ((-1) * eqcol - 1 + int(numberOfDof))] =  aux

    return(kt)

"Vector of forces related to the nodes that are free"
def calculateFp(EL, NL, force):
    equations = getEquations(NL)
    numberOfDof = np.max(equations)
    auxDouble = np.zeros(int(numberOfDof))
    
    "creating the total force vector"
    force_total = np.zeros([np.size(NL, 0), np.size(NL, 1)])
    for i in range (0, np.size(force, 0)):
        aux = force[i, 0]
        force_total [aux -1, 0] = force[i , 1]
        force_total [aux -1, 1] = force[i , 2]
    
    "Loop to create the load vector of the free nodes"
    for j in range(0, np.size(NL, 0)): 
        node_eq = equations[j]
        for k in range(0, np.size(NL, 1)):
            auxEq = node_eq[k]
            if auxEq > 0:
                auxDouble[(int(auxEq) - 1)] = force_total[j, k]
      
    nodalForces = auxDouble
    return nodalForces

"Function that changes the Coordenates on the variable NL"
def setDeformedCoordenates(NL, displacement):
    equations = getEquations(NL).flatten()
    NL_f = NL.flatten()
    
    "Scale to see the result"
    scale = 2000
    
    count = 0
    for i in range(0, np.size(NL_f,0)):
        eq_nodal = equations[i]
        if eq_nodal > 0:
            NL_f[i] += displacement[count] * scale
            count +=1
    
    NL_f = NL_f.reshape(np.size(NL,0), np.size(NL,1))
    return NL_f

"Function that creates the full vector of displcaments of the model"
def setTotalDisplacement(NL, displacement):
    equations = getEquations(NL).flatten()
    results = np.zeros(np.size(equations))
    
    count = 0
    for i in range(0, np.size(equations,0)):
        eq_nodal = equations[i]
        if eq_nodal > 0:
            results[i] = displacement[count]
            count +=1
    
    results = results.reshape(np.size(NL,0), np.size(NL,1))
    return results

"Function responsable to solve the linear equation system and ploting the results"
def solverModel(EL, NL, force):
    plotMesh(EL, NL, 'red', True)
    
    Kuu = calculateKuu(EL,NL)
    fp =  calculateFp(EL, NL, force)
    displacement = np.linalg.solve(Kuu, fp)
    
    NL_def = setDeformedCoordenates(NL, displacement)
    plotMesh(EL, NL_def, 'blue', True)
    results = setTotalDisplacement(NL, displacement)
    
    return(results, NL_def)    



"""
POST-PROCESSING
"""
"Function that calculates the strain"
def calculateStrain(EL, displacement):
    total_strain = []
    
    for i in range(0,np.size(EL,0)):
        x = np.zeros([np.size(EL,1),2])
        nodes = EL[i]
       
        "Loop to get the coordenates of each node that make the element and the equation."
        for j in range(0, len(nodes)):
            x[j] = NL[int(nodes[j]-1)]
            
        "Loop to get the displacement of each node of the element"
        qe = np.zeros([4,2])
        for k in range(0, np.size(EL, 1)):
            node = EL[i, k]
            for n in range(0,2):
                qe[k,n] = displacement[int(node)-1,n]
        qe = qe.flatten()
        
        NPE = np.size(x,0) #nodes per element
        coord = x.T #coordinates of the nodes
        #we take the matrix B in eta and xi = 0, if not we would have 4 stress results
        grad_N_Nat = defineGrad_N_Nat(NPE, 0, 0)
        J = defineJ(coord, grad_N_Nat)
        DG = defineDG(J, grad_N_Nat) 
        B = defineB(DG)
        strain = B @ qe
        total_strain.append(strain)
    total_strain = np.array(total_strain)
    return total_strain

"Function that calculates the stress"
def calculateStress(E, mu, strain):
    D = defineConstitutiveD(E, mu)
    stress_total = []
    for i in range(0, np.size(strain,0)):
        stress_total.append(D @ strain[i])
    
    stress_total = np.array(stress_total)
    return stress_total



"""
SOLVING THE PROBLEM:
"""
(NL,EL) = maillage(d1, d2, p, m)
K = calculateK(EL, NL)
displacement, NL_def = solverModel(EL, NL, force)
strain = calculateStrain(EL, displacement)
stress = calculateStress(E, mu, strain)


"""
PRINTING THE RESEULTS
"""
# print('RESULTS:')
# print('================================================================')
# print('========================================')
# print('List of elements:\n', 'N1', '', 'N2', '', 'N3', '', 'N4')
# print(EL)
# print('========================================')
# print('Nodes coordinates:\n', ' x' , '    ', 'y')
# print(np.around(NL, 4))
# print('========================================')
# print('Global stiffness matrix:\n', '  x' , '             ', 'y')
# print(np.around(K,2))
# print('========================================')
# print('Displacementof the nodes:\n', '  x' , '             ', 'y')
# print(displacement)
# print('========================================')
# print('Strain on the elements:\n', ' Epsilon_xx', '    ', 'Epsilon_yy', '    ', 'eEpsilon_xy')
# print(strain)
# print('========================================')
# print('Stress on the elements:\n', ' Sigma_xx', '    ', 'Sigma_yy', '    ', 'Sigma_xy')
# print(np.around(stress/10**(6), 2))
# print('========================================')
# print('================================================================')


"""
WRITING THE RESULTS ON A TXT
"""
def wrintingResults(displacement, strain, stress, K):
    
    f = open("output.txt", 'w')
    f.write('RESULTS:\n')
    f.write('================================================================\n')
    f.write('========================================\n')
    f.write('List of elements:\n' + 'N1' + '   ' + 'N2' + '   ' + 'N3'+ '   ' + 'N4\n')
    f.write(str(EL) + '\n')
    f.write('========================================\n')
    f.write('Nodes coordinates:\n' + ' x' + '    ' + 'y\n')
    f.write(str(np.around(NL, 4)) + '\n')
    f.write('========================================\n')
    f.write(str(np.around(K,2)) + '\n')
    f.write('========================================\n')
    f.write('Displacementof the nodes:\n' + '  x' + '             '+ 'y\n')
    f.write(str(displacement) + '\n')
    f.write('========================================\n')    
    f.write('Strain on the elements:\n' + ' Epsilon_xx' + '    ' + 'Epsilon_yy' + '    ' + 'Epsilon_xy\n')
    f.write(str(strain) + '\n')
    f.write('========================================\n')    
    f.write('Stress on the elements:\n' + ' Sigma_xx' + '    ' + 'Sigma_yy' + '    ' + 'Sigma_xy\n')
    f.write(str(np.around(stress/10**(6), 2)) + '\n')
    f.write('========================================\n')
    f.write('================================================================\n')
    f.close()  

wrintingResults(displacement, strain, stress, K)
