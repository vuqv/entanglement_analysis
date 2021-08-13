#!/usr/bin/python

# Dependencies
import os, getopt
import sys
from math import sqrt, pi
import time


def main(argv):
    #Argument
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('script.py -i <inputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('script.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    print('Input file is ', inputfile) 


    # Open PDB inpput files
    fCA = open("ca_res.dat", "w")
    fPDB = open(inputfile, "r")

    # Extract coordinates from PDB file
    xCA = []; yCA = []; zCA = []

    if fPDB.mode == "r":
        lines = fPDB.readlines()
        NATM = len(lines) - 1
        for i in range(NATM):
            line = lines[i].split()
            if line[2] == "CA":
                fCA.write(lines[i])
                xCA.append(float(line[6]))
                yCA.append(float(line[7]))
                zCA.append(float(line[8]))

    fCA.close()
    fPDB.close()

    
    # Identify loop
    fLOOP = open("loop_index.dat","w")
    NCA = len(xCA)
    x_l1 = []; y_l1 = []; z_l1 = []; x_l2 = []; y_l2 = []; z_l2 = []
    ndx_i1 = []; ndx_i2 = []; ndx_j1 = []; ndx_j2 = []

    for i1 in range(NCA-10):
        for i2 in range(i1+10, NCA):
            distCA = sqrt((float(xCA[i2]) - float(xCA[i1]))**2 + (float(yCA[i2]) - float(yCA[i1]))**2 + (float(zCA[i2]) - float(zCA[i1]))**2)
            for j1 in range(NCA-10):
                for j2 in range(j1+10, NCA):
                    if (distCA <= 9.0 and ((j1 < i1 and j2 < i1) or (j1 > i2 and j2 > i2))):
                        fLOOP.write("%5.0f %5.0f %10.3f %5.0f % 5.0f\n" % (i1+1, i2+1, distCA, j1+1, j2+1))
                        ndx_i1.append(i1)
                        ndx_i2.append(i2)
                        ndx_j1.append(j1)
                        ndx_j2.append(j2)

    fLOOP.close()

    fGE = open("gaussian_entanglement.dat", "w")
    fmaxGE = open("log.log", "w")
    Nloop = len(ndx_i1)
    maxGE = 0.0
    for x in range(Nloop):
        GE = 0.0
        for y in range(ndx_i1[x], ndx_i2[x]):
            xav_l1 = 0.5*(xCA[y] + xCA[y+1])
            yav_l1 = 0.5*(yCA[y] + yCA[y+1])
            zav_l1 = 0.5*(zCA[y] + zCA[y+1])
            dx_l1 = xCA[y+1] - xCA[y]
            dy_l1 = yCA[y+1] - yCA[y]
            dz_l1 = zCA[y+1] - zCA[y]
            for z in range(ndx_j1[x], ndx_j2[x]):
                xav_l2 = 0.5*(xCA[z] + xCA[z+1])
                yav_l2 = 0.5*(yCA[z] + yCA[z+1])
                zav_l2 = 0.5*(zCA[z] + zCA[z+1])
                dx_l2 = xCA[z+1] - xCA[z]
                dy_l2 = yCA[z+1] - yCA[z]
                dz_l2 = zCA[z+1] - zCA[z]
                part_Ax = xav_l1 - xav_l2
                part_Ay = yav_l1 - yav_l2
                part_Az = zav_l1 - zav_l2
                part_B = (sqrt((part_Ax)**2 + (part_Ay)**2 + (part_Az)**2))**3
                part_Cx = (dy_l1*dz_l2 - dz_l1*dy_l2)
                part_Cy = (dz_l1*dx_l2 - dx_l1*dz_l2)
                part_Cz = (dx_l1*dy_l2 - dy_l1*dx_l2)
                GE = GE + (0.25/pi)*((part_Ax*part_Cx) + (part_Ay*part_Cy) + (part_Az*part_Cz))/part_B
        fGE.write("%5.0f %5.0f %5.0f %5.0f %10.3f\n" % (ndx_i1[x]+1, ndx_i2[x]+1, ndx_j1[x]+1, ndx_j2[x]+1, GE))
        if abs(GE) > maxGE:
            maxGE = abs(GE)
            fmaxGE.write("%5.0f %5.0f %5.0f %5.0f %10.3f\n" % (ndx_i1[x]+1, ndx_i2[x]+1, ndx_j1[x]+1, ndx_j2[x]+1, maxGE))       

    print('The largest absolute value of the mutual entanglement is ', maxGE)

    fGE.close()
    fmaxGE.close()
    return


if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print("%s %f" % ("Runtime: ",end - start))



# To use: python3.8 script.py -i <pdb file>
