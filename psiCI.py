# BSD 2-Clause License
# 
# Copyright (c) 2025, Francois Mauger
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Configuration-interaction module using Psi4 with DFT/HF orbitals
# 
# Reference: G. Visentin and F. Mauger, "Configuration-interaction calculations 
#            with density-functional theory molecular orbitals for modeling 
#            valence- and core-excited states in molecules," arXiv:2509.08245
#            (2025)

import numpy as np
import psi4
import scipy.sparse.linalg as scp
import itertools as it

## Miscellaneous methods ===========================================================
def au2ev(val):
    """
    Convert energy from atomic units to electron volts
        1 [a.u.] = 27.211386245988 [eV]

    Parameters:
    -----------
    auValue : float or numpy.array
        Atomic unit energy(ies) to be converted to electron volts

    Returns:
    --------
    evValue : float or numpy.array
        Converted energy(ies) in electron volts
    """

    return val*27.211386245988

def ev2au(val):
    """
    Convert energy from electron volts to atomic units
        1 [eV] = 1/27.211386245988 [a.u.]

    Parameters:
    -----------
    evValue : float or numpy.array
        Electron volt energy(ies) to be converted to atomic units

    Returns:
    --------
    auValue : float or numpy.array
        Converted energy(ies) in atomic units
    """

    return val/27.211386245988
    
## CI class ========================================================================
class psiCI:
    """
    Configuration interaction module using Psi4 with DFT/HF orbitals
        Use psiCI to describe a configuration-interaction (CI) model associated with
        a spin-restricted set of spatial orbitals, typically obtained via a density-
        functional theory (DFT) or Hartree-Fock (HF) ground state calculation in 
        Psi4. The class also provides functionalities for calculating the CI matrix 
        and associated ground- and excited-state wave functions.

    Attributes
    ----------
    waveFunction : psi4.core.RHF
        Wave function object returned by Psi4 after a Hartree-Fock or Density-
        functional theory calculation
    
    numberElectron : int (default -1)
        Number of electrons in the model. If negative, the number of electrons is 
        recovered from the waveFunction object input.

    configuration : numpy.array (default np.empty(0)
        Configuration state basis, defined as a numpy array of waveFunction indexes.
        Each row specifies the indexes of the spatial orbitals making up the associated 
        configuration state, with positive (resp. negative) indexes specifying up
        (resp. down) spin orbitals. For instate [-1, -2, 1, 2] corresponds to the 
        Slater determinant formed by the two deepest-energy spin orbitals in each of
        the up and down spin channels.
        
    display : bool (default True)
        Whether to display the progress of CI calculation. 
        
    tolerance : float (default 1e-10)
        Threshold for setting CI matrix elements to zero. Set tolerance to a negative
        value to disable the thresholding.

    Methods
    -------
    showDocumentation : displays the run-time documentation
        The run-time documentation prints out the parameters for the configuration-
        interaction model object.

    computeCI : Compute the configuration-interaction matrix
        Use computeCI to calculate the CI matrix, associated with the configuration-
        states basis, in the orbital basis.

    setConfiguration : Set the configuration state basis
        Use setConfiguration to build the configuration-state basis for CI calculation.

    analyzeSpectrum : Analyze the spectrum of a CI matrix
            Use analyzeSpectrum to display the spectrum analysis of a CI matrix.
    """
    
    # Version information and control ==============================================
    __author__       = "G. Visentin & F. Mauger"
    __version__      = "00.01.004"
    __lastModified__ = "04/27/2025"

        # Change log
        # ----------   ---------   -------------------------------------------------
        #   Date     |  Version  |  Author
        # ----------   ---------   -------------------------------------------------
        # 03/02/2025 | 00.01.000 | F. Mauger
        #    Initiate version control number
        # 03/03/2025 | 00.01.001 | F. Mauger
        #  > Fix __getExcitationIndex (using deep copy to prevent messing with 
        #    configuration basis)
        #  > Add energy values in eV in analyzeSpectrum
        #  > Add CISD to setConfiguration
        #  > Fix trivially vanishing terms in CI matrix (difference by more than two 
        #    spin orbitals)
        #  > Add progress bar for building CI matrices with more than 500 
        #    configuration states in the basis
        #  > Optimize the eigensolver to use (eigh vs eigsh depending on the size of
        #    the CI matrix and indexes of the excited states requested in the spectrum
        #    analysis.
        # 03/04/2025 | 00.01.002 | F. Mauger
        #  > Add option to calculate and return the dipole matrices
        #  > Fix analyzeSpectrum for large matrix and only showing the ground state
        # 04/26/2025 | 00.01.003 | F. Mauger
        #  > Add cleanConfiguration
        #  > Add frozen, noDouble, and noEmpty options for setConfiguration
        #  > Add support for multi-reference CIS(D)
        # 04/27/2025 | 00.01.004 | F. Mauger
        #  > Add support for RAS
    
    # Class creation and initialization ============================================
    def __init__(self,waveFunction:psi4.core.RHF,numberElectron:int=-1,configuration:np.array=np.empty(0),display:bool=True,tolerance:float=1e-10):
        # Copy input parameters
        self.waveFunction       = waveFunction
        if numberElectron > 0:
            self.numberElectron = numberElectron                         # user-specified number of electrons
        else:
            self.numberElectron = 2*waveFunction.nalpha()                # read from orbital
        self.configuration      = configuration
        self.display            = display
        self.tolerance          = tolerance

    # Run-time documentation =======================================================
    def showDocumentation(self,isRun:bool=False):
        """
        Displays the run-time documentation
            The run-time documentation prints out the parameters for the configuration-
            interaction model object.
        """
        self.__showHeader()
        
        print("=== Electronic structure =======================================================")
        print("  * " + str(self.numberElectron) + " electrons")
        print("  * " + str(self.waveFunction.Ca().np.shape[1]) + " spatial orbitals")
        if self.configuration.shape[0] > 0:
            print("  * " + str(self.configuration.shape[0]) + " configurations\n")
        else:
            print(" ")

        print("=== Configuration-interaction calculation ======================================")
        if self.tolerance > 0:
            print("  * tolerance: " + str(self.tolerance))
        
        if not isRun:
            self.__showFooter()
        
    # Display the header ===========================================================
    def __showHeader(self):
        print("################################################################################")
        print("##       Configuration-interaction module for Psi4 with DFT/HF orbitals       ##")
        print("################################################################################")
        print("  * Author(s):     " + self.__author__)
        print("  * Version:       " + self.__version__)
        print("  * Last modified: " + self.__lastModified__ + "\n")

    # Display the footer ===========================================================
    def __showFooter(self):
        print("\n################################################################################\n")
    
    # Compute the CI matrix ========================================================
    def computeCI(self,returnDipole:bool=False):
        """
        Compute the configuration-interaction matrix
            Use computeCI to calculate the CI matrix, associated with the configuration-
            states basis, in the orbital basis.

        Parameters
        ----------
        returnDipole : bool (default False)
            Whether to calculate and return the dipole transition matrices in the x, y, 
            and z directions alongside the CI matrix

        Output variables
        ----------------
        CI : numpy.array
            Matrix containing the CI elements associated with the member orbital and 
            configuration state bases.
            Note: the CI matrix elements only contain the electronic part to the energy
            and any nuclear repulsion component must be added afterward.

        Dx : numpy.array
            Dipole-transition matrix in the x direction. It is only returned when 
            returnDipole is True

        Dy : numpy.array
            Dipole-transition matrix in the y direction. It is only returned when 
            returnDipole is True

        Dz : numpy.array
            Dipole-transition matrix in the z direction. It is only returned when 
            returnDipole is True
        """
        #Note: Psi4 mints.mo_eri uses the chemists' notation for the two-electron integrals

        # Initialization ~~~~~~~~~~~~~~~~~~~
        if self.display:
            self.showDocumentation(True)

        mints = psi4.core.MintsHelper(self.waveFunction.basisset())                 # Psi4 mints helper
        iMO   = np.unique(np.abs(self.configuration))                               # index of used spatial orbitals; use iMO to map spin-orbital indexes to 
                                                                                    # core and electron-repulsion integral array elements
        if all(iMO == np.asarray(range(1,len(iMO)+1),dtype=int)):                   # mapping spatial-orbital indexes to the used ones
            iiMO = iMO                                                              # no skipped orbitals => same orbital indexes
        else:
            iiMO = np.full(iMO[-1],iMO[-1]+100,dtype=int)                           # map each orbital index individually (put unmapped index out of range)
            for k in range(iMO[-1]):
                ind = np.where(iMO == (k+1))[0]                                     # check for used index
                if len(ind) > 0:
                    iiMO[k] = ind[0] + 1                                            # map used index

        if self.display:
            print("  * " + str(len(iMO)) + " active spatial orbitals\n")

        # Core Hamiltonian ~~~~~~~~~~~~~~~~~
        if self.display:
            print("=== CI matrix calculation ======================================================")
            print("  * Core Hamiltonian",end="")
            
        C = np.asarray(self.waveFunction.Ca())
        H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())       # core Hamiltonian in the AO basis
        H = C.T @ H @ C                                                             # convert to the MO basis
        H = H[:,iMO-1][iMO-1,:]                                                     # select active MOs

        if self.tolerance > 0:                                                      # clean near-zero elements
            H[abs(H) < self.tolerance] = 0
        
        if self.display:
            print("                                                        done")

        # Dipole coupling matrices ~~~~~~~~~
        if returnDipole:
            if self.display:
                print("  * Dipole coupling",end="")

            D    = np.asarray(mints.ao_dipole())                                    # dipole-coupling matrices
            D[0] = C.T @ D[0] @ C                                                   # convert to the MO basis
            D[1] = C.T @ D[1] @ C
            D[2] = C.T @ D[2] @ C
            D    = D[:,:,iMO-1][:,iMO-1,:]                                          # select active MOs

            if self.tolerance > 0:                                                  # clean near-zero elements
                D[abs(D) < self.tolerance] = 0
            
            if self.display:
                print("                                                         done")
        
        # Two-electron integrals ~~~~~~~~~~~
        if self.display:
            print("  * Two-electron integrals",end="")
        C   = psi4.core.Matrix.from_array(C[:,iMO-1])
        ERI = np.asarray(mints.mo_eri(C,C,C,C))

        if self.tolerance > 0:                                                      # clean near-zero elements
            ERI[abs(ERI) < self.tolerance] = 0
            
        if self.display:
            print("                                                  done")

        # Build the CI matrix ~~~~~~~~~~~~~~
        nCS = self.configuration.shape[0]
        CI  = np.zeros((nCS,nCS),dtype=float)                                       # CI matrix
        if returnDipole:
            Dx = np.zeros((nCS,nCS),dtype=float)                                    # dipole coupling matrices
            Dy = np.zeros((nCS,nCS),dtype=float)
            Dz = np.zeros((nCS,nCS),dtype=float)
        
        disp = False                                                                # by default do not display the progress bar
        if self.display:
            print("  * CI matrix",end="",flush=True)
            if nCS > 500:
                print("                |",end="")                                   # initialize the progress bar
                disp = True
                ndp  = nCS**2                                                       # how many CI elements to calculate
                idp  =   1;                                                         # how many progress bars displayed
                ikl  =   0;                                                         # how many CI elements calculated

        for k in range(nCS):                                                        # parse through configuration states
            iMOk = iiMO[abs(self.configuration[k,:])-1]                             # map MO index to precaculated element indexes
            iMOk = iMOk*np.sign(self.configuration[k,:])                            # restore spin information
            n        = np.abs(iMOk)-1                                               # spatial orbital indexes

            # Diagonal element
            CI[k,k]  = np.sum(H[n,n])                                               # core component sum_a <a|h|a>
            if returnDipole:                                                        # dipole coupling: sum_a <a|x,y,z|a>
                Dx[k,k] = np.sum(D[0,n,n])
                Dy[k,k] = np.sum(D[1,n,n])
                Dz[k,k] = np.sum(D[2,n,n])
                
            for l in iMOk:
                al       = np.abs(l)-1
                CI[k,k] += 0.5*np.sum(ERI[al,al,n,n])                               # two-electron components sum_nm <nm|nm> = [nn|mm] (spin always match)
                CI[k,k] -= 0.5*np.sum(ERI[al,n,n,al]*(l*iMOk > 0))                  # two-electron components sum_nm <nm|mn> = [nm|mn] (check matching spin)

            if disp:                                                                # update the progress bar
                ikl += 1
                if np.round(50.0*ikl/ndp) > idp:
                    print("|",end="",flush=True)
                    idp += 1

            # Off-diagonal elements
            for l in range(k+1,nCS):
                # Get relative excitation indexes
                a, r, s = self.__getExcitationIndex(self.configuration[k,:],self.configuration[l,:])

                # Calculate matrix element
                if len(a) == 1:
                    ia = iiMO[abs(a)-1]-1
                    ir = iiMO[abs(r)-1]-1
                    if a*r > 0:                                                     # the two configuration states have the same net spin
                        CI[k,l]  = H[ia,ir]                                         # core component <a|h|r>
                        if returnDipole:                                            # dipole coupling <a|x,y,z|r>
                            Dx[k,l] = D[0,ia,ir]
                            Dy[k,l] = D[1,ia,ir]
                            Dz[k,l] = D[2,ia,ir]
                        CI[k,l] += np.sum(ERI[ia,ir,n,n])                           # two-electron components sum_n <an|rn> = [ar|nn]
                        CI[k,l] -= np.sum(ERI[ia,n,n,ir]*(a*iMOk > 0))              # two-electron components sum_n <an|nr> = [an|nr]

                elif len(a) == 2:
                    ia = iiMO[abs(a)-1]-1
                    ir = iiMO[abs(r)-1]-1
                    if all(a*r > 0):                                                # two-electron component <ab|rs> = [ar|bs]
                        CI[k,l]  = ERI[ia[0],ir[0],ia[1],ir[1]]
                    if all(a*r[[1,0]] > 0):                                         # two-electron component <ab|sr> = [as|br]
                        CI[k,l] -= ERI[ia[0],ir[1],ia[1],ir[0]]

                # Round-off correction, sign, and symmetry
                if abs(CI[k,l]) < self.tolerance:                                   # force matrix element to zero
                    CI[k,l] = 0
                else:
                    CI[k,l] = s*CI[k,l]                                             # adjust for sign of the permutation
                    CI[l,k] = CI[k,l]                                               # use symmetry (Psi4 uses real orbitals)

                if returnDipole:                                                    # same procedure for the dipole elements as CI
                    if abs(Dx[k,l]) < self.tolerance:
                        Dx[k,l] = 0
                    else:
                        Dx[k,l] = s*Dx[k,l]
                        Dx[l,k] = Dx[k,l]
                        
                    if abs(Dy[k,l]) < self.tolerance:
                        Dy[k,l] = 0
                    else:
                        Dy[k,l] = s*Dy[k,l]
                        Dy[l,k] = Dy[k,l]
                        
                    if abs(Dz[k,l]) < self.tolerance:
                        Dz[k,l] = 0
                    else:
                        Dz[k,l] = s*Dz[k,l]
                        Dz[l,k] = Dz[k,l]
                    
                if disp:                                                            # update the progress bar
                    ikl += 2
                    if np.round(50.0*ikl/ndp) > idp:
                        print("|",end="",flush=True)
                        idp += 1
        
        if self.display:                                                            # CI matrix built => update display
            if disp:
                if idp < 51:                                                        # somehow short of a progress bar
                    print("|")
                else:
                    print(" ")
            else:
                print("                                                               done")
            
        # Finalization
        if self.display:
            self.__showFooter()

        if returnDipole:
            return CI, Dx, Dy, Dz
        else:
            return CI

    # Identify the relative excitation indexes =====================================
    def __getExcitationIndex(self,CS1in,CS2in):
        """
        Relative excitation indexes and sign
            Use __getExcitationIndex to indentify the indexes a in |CS1> and r in |CS2> such
            that the configuration state |CS2> = s*|CS1_a^r>, with the scalar s (+/-1) 
            corresponding to the sign of the permutation. If the two configuration states are
            identical or differ by more than two excitations, the output are all set to empty
            vectors.

        Parameters
        ----------
        CS1 : numpy.array
            Indexes of the spin orbitals defining the configuration state |CS1>
            
        CS2 : numpy.array
            Indexes of the spin orbitals defining the configuration state |CS2>

        Output variables
        ----------------
        a : numpy.array
            Excitation index for |CS1>
            
        b : numpy.array
            Excitation index for |CS2>

        s : 1 or -1
            Sign of the permutation
        """

        # Initialization
        CS1 = np.copy(CS1in)                                                        # changing CS1 and CS2 below but we don't want it
        CS2 = np.copy(CS2in)                                                        # reflected in the configuration state basis
        C, ic, ie = np.intersect1d(CS1,CS2,assume_unique=True,return_indices=True)

        if (len(C) == len(CS1)): 
            raise RuntimeError("Input configuration states shouldn't be matching")  # unexpected error
            
        elif (len(C) < len(CS1)-2):                                                 # vanishing two-electron integrals (don't bother calculating a and r)
            a = np.empty(0)
            r = np.empty(0)
            s = 0

        else:                                                                       # non trivially vanishing two-electron intergal
            # Align common components
            s = 1

            for k in ic:                                                            # align common spin orbital indexes (for the sign)
                n = np.where(CS2 == CS1[k])[0]
                if n != k:                                                          # swap spin orbitals to align
                    CS2[n], CS2[k] = CS2[k], CS2[n]
                    s *= -1
                    
            if not all(CS1[ic] == CS2[ic]):
                raise RuntimeError("Common spin orbital indexes not properly aligned. Contact developers.")

            # Excitation indexes
            ie = np.setdiff1d(range(len(CS1)),ic,assume_unique=True)
            a = CS1[ie]                                                             # relative excitation indexes
            r = CS2[ie]
        
        return a, r, s
    
    # Build configuration state basis ==============================================
    def setConfiguration(self,reference:np.array=np.empty(0),mode:str="CIS",active:np.array=np.empty(0),
                         frozen:np.array=np.empty(0),noDouble:np.array=np.empty(0),noEmpty:np.array=np.empty(0)):
        """
        Set the configuration state basis
            Use setConfiguration to build the configuration-state basis for CI calculation.

        Parameters
        ----------
        reference : numpy.array (default numpy.empty(0))
            For CIS and CISD calculations, reference state configuration, defined as a numpy
            vector of orbital indexes, with positive (resp. negative) indexes specifying up
            (resp. down) spin orbitals. For instate [-1, -2, 1, 2] corresponds to the Slater
            determinant formed by the two deepest-energy spin orbitals in each of the up and
            down spin channels. If reference is left empty, a single reference singlet (for
            even numberElectron) or doublet (odd, with the unparied electron in the up-spin
            channel) state composed of the self.numberElectron deepest spin orbitals is used
            for the reference. If a reference is specified, self.numberElectron is updated
            to reflect the number of electrons associated with the reference.

            For multi-reference CIS and CISD calculations, specify the different reference 
            configurations in separate row of the reference matrix. For instance, using
            [[-1, -2, 1, 2],[-1, -3, 1, 3]] corresponds to the first reference state with 
            two electron in the first two spatial orbitals and the second reference with two
            electrons in the first and third spatial orbitals.

            For RAS calculations, the reference is used to determine the number of electrons
            in each spin channel and has no other bearing in the determination of the active
            space (i.e., it may not be included in the configuration state basis). For
            instance [-1, -2, 1, 2, 3, 4] specifies a triplet state with two unpaired electrons 
            in the up spin channel. If left empty, a singlet (for even numberElectron) or
            doublet (odd, with the unparied electron in the up-spin channel) configuration is
            used. If a reference is specified, self.numberElectron is updated to reflect the
            number of electrons associated with the reference.

        mode : str (default "CIS")
            Mode of CI calculation for which to set the configuration-state basis:
              * "CIS" sets the configuration basis for single-excitation type CI (CIS).
              * "CISD" sets the configuration basis for single- and double-excitation type
                CI (CISD).
              * "RAS" sets the configuration basis for restricted active space type CI (RAS).
                If no noDouble or noEmpty are specified, this corresponds to a complete
                active space (CAS) model.
              
        active : numpy.array (default numpy.empty(0))
            Indexes of the spin orbitals to use in the configuration space expansion, with
            positive (resp. negative) indexes specifying up (resp. down) spin orbitals. For
            instance, [-1, -2, -3, -4, 1, 2, 3] indicates that the spin orbitals formed by
            the first 4 spatial orbitals with down spin and only the first 3 spatial orbitals
            with up spin should be considered for building the configuration state basis. If
            left empty, all spin orbitals are included in the basis, thus corresponding to
            full CIS(D)/RAS/CAS.

            For multi-reference CIS(D) calculations, all the reference share the same active
            space.

        frozen : numpy.array (default numpy.empty(0))
            Indexes of the frozen spin orbitals, which must contain one electron. For
            instance [-1, 1, 2] indicates all configuration states in the configuration state
            basis must include the spin orbitals formed by the first spatial orbitals with
            down spin and the first 2 spatial orbitals with up spin. If left empty, no orbitals
            are frozen.
            
            For single- and multi-reference CIS and CISD calculations, only frozen orbitals
            that are in the reference are considered.

            For RAS calculations, the number of frozen orbitals in each spin channel must be
            compatible with the number of electrons each hold (defined via the reference).

        noDouble : numpy.array (default numpy.empty(0))
            Indexes of the spatial orbitals that may not contain two electrons, i.e., have an
            electron in both the up and down spin channels. For instance, [5, 6] indicates that
            the spatial orbitals 5 and 6 may not be fully occupied. If left empty, no restriction
            on double occupation is imposed.
            
            For single- and multi-reference CIS and CISD calculations, the noDouble constraint
            is NOT imposed on the reference configuration(s).

        noEmpty : numpy.array (default numpy.empty(0))
            Indexes of the spatial orbitals that may not be left empty, i.e., must have at 
            least one electron in the up or down spin channel. For instance [1, 2], indicates that
            the first two spatial orbitals should always hold at least one electron. If left empty,
            no restriction on empty occupation is imposed.
            
            For single- and multi-reference CIS and CISD calculations, the noEmpty constraint
            is NOT imposed on the reference configuration(s).
        """

        # Initialization
        if self.display:
            self.__showHeader()
            print("=== Configuration-state basis ==================================================")

        if (len(reference) == 0):
            nbRef     = 1
            
            reference = np.arange(1,np.floor(self.numberElectron/2)+1,dtype=int)
            reference = np.concatenate((-reference,reference))
            
            if len(reference) < self.numberElectron:
                reference = np.concatenate((reference,reference[-1]+1),axis=None)   # triplet state, unpaired electron in up spin channel
                
        else:
            self.numberElectron = reference.shape[-1]
            if self.numberElectron == len(reference):
                nbRef = 1
            else:
                nbRef = np.size(reference,0)

        if len(active) == 0:
            active = np.arange(1,self.waveFunction.Ca().np.shape[0]+1,dtype=int)
            active = np.concatenate((-active,active))

        if mode.lower() == 'ras':
            algo = 0
        elif mode.lower() == "cis":
            algo = 1
        elif mode.lower() == "cisd":
            algo = 2
        else:
            raise RuntimeError("Unknown configuration mode " + mode)

        # Display configuration-state basis parameters
        if self.display:
            print("  * Type        = " + mode.upper())
            print("  * Nb. elec.   = " + str(self.numberElectron))
            
            if algo > 0:
                if nbRef == 1:
                    print("  * Total spin  = " + str(0.5*np.sum(np.sign(reference))))
                    print("  * Reference   = " + str(reference))
                else:
                    print("  * Total spin  = " + str(0.5*np.sum(np.sign(reference),axis=1)))
                    print("  * References  = " + str(reference[0]))
                    for k in range(1,nbRef):
                        print("                  " + str(reference[k]))
            else:
                print("  * Total spin  = " + str(0.5*np.sum(np.sign(reference))))

            print("  * Active      = " + str(active[active < 0]))
            print("                = " + str(active[active > 0]))
            
            if len(frozen) > 0:
                print("  * Frozen      = " + str(frozen))
                
            if len(noDouble) > 0:
                print("  * No double   = " + str(noDouble))
                
            if len(noEmpty) > 0:
                print("  * Not empty   = " + str(noEmpty))

        # Build the configuration-state basis
        if algo > 0: # CIS(D) calculation
            # Copy the reference(s)
            self.configuration = reference

            # Add single excitations
            if nbRef == 1:
                self.configuration = np.concatenate(([self.configuration],
                                                     self.__configurationCIS(reference,active,frozen,noDouble,noEmpty)))
            else:
                for k in range(nbRef):
                    self.configuration = np.concatenate((self.configuration,
                                                     self.__configurationCIS(reference[k],active,frozen,noDouble,noEmpty)))

            # Add double excitations
            if algo > 1:
                if nbRef == 1:
                    self.configuration = np.concatenate((self.configuration,
                                                     self.__configurationCID(reference,active,frozen,noDouble,noEmpty)))
                else:
                    for k in range(nbRef):
                        self.configuration = np.concatenate((self.configuration,
                                                     self.__configurationCID(reference[k],active,frozen,noDouble,noEmpty)))

            # Remove duplicates
            if nbRef > 0:
                self.cleanConfiguration()

        elif algo == 0:
            # Initialization
            act = np.setdiff1d(active,frozen)
            actP = act[act > 0]
            actN = act[act < 0]

            frz = np.intersect1d(frozen,active)
            nbP = len(reference[reference > 0]) - len(frz[frz > 0])                 # number of active electrons in spin channels
            nbN = len(reference[reference < 0]) - len(frz[frz < 0])

            # RAS subspaces
            rasP = np.fromiter(it.combinations(actP, nbP), dtype=np.dtype((int, nbP)))
            rasN = np.fromiter(it.combinations(actN, nbN), dtype=np.dtype((int, nbN)))
            nCSB = len(rasP) * len(rasN)

            self.configuration = np.empty((nCSB,self.numberElectron),dtype=int)

            # Build the RAS basis
            n = 0

            if len(frz[frz < 0]):                                                   # down spin frozen orbitals
                self.configuration[:,n:n+len(frz[frz < 0])] = np.tile(frz[frz < 0], (nCSB,1))
                n += len(frz[frz < 0])

            if nbN > 0:                                                             # down spin active space
                self.configuration[:,n:n+nbN] = np.tile(rasN, (len(rasP),1))
                n += nbN

            if len(frz[frz > 0]):                                                   # up spin frozen orbitals
                self.configuration[:,n:n+len(frz[frz > 0])] = np.tile(frz[frz > 0], (nCSB,1))
                n += len(frz[frz > 0])

            if nbP > 0:                                                             # up spin active space
                for k in range(len(rasN)):
                    self.configuration[k::len(rasN),n:n+nbP] = rasP

            # Impose constraints
            self.configuration = self.__constrainedConfiguration(self.configuration,noDouble,noEmpty)
        
        else:
            raise RuntimeError("Unexpected error building the configuration basis. Contact a developer")
            
        # Finalize
        if self.display:
            print("  * " + str(np.size(self.configuration,0)) + " configurations")
            self.__showFooter()

    # Constrained configuration state ==============================================
    def __constrainedConfiguration(self,CSB,noDouble,noEmpty):
        """
        Removes configuration states that do not fulfil the noDouble and noEmpty constraints
        """

        # Initialization
        if (len(noDouble) > 0) or (len(noEmpty) > 0):
            ind = np.full(np.size(CSB,0),True)

        # Constraints
        for k in range(len(noDouble)):                                              # double excitations                                  
            ind *= np.logical_not(np.any(CSB == noDouble[k],axis=1) * np.any(CSB ==-noDouble[k],axis=1))

        for k in range(len(noEmpty)):                                               # empty spatial orbital
            ind *= np.logical_not(np.all(CSB != noEmpty[k],axis=1) * np.all(CSB !=-noEmpty[k],axis=1))

        if (len(noDouble) > 0) or (len(noEmpty) > 0):
            CSB = CSB[ind]

        return CSB

    # Configuration state for single excitations ===================================
    def __configurationCIS(self,reference,active,frozen,noDouble,noEmpty):
        """
        Configuration state basis for single reference single excitation
        """

        # Initialization
        act = np.setdiff1d(active,reference)
        actP = act[act > 0]
        actN = act[act < 0]
        
        frz, _, iFrz = np.intersect1d(frozen,reference,return_indices=True)

        nCSB = (np.sum(reference < 0)-np.sum(frz < 0))*len(actN) + (np.sum(reference > 0)-np.sum(frz > 0))*len(actP)
        CSB  = np.tile(reference, (nCSB,1))

        # Parse through single excitations
        n = 0
        for k in range(len(reference)):
            if all(iFrz != k):
                if reference[k] > 0:
                    CSB[n:n+len(actP),k] = actP
                    n += len(actP)
                else:
                    CSB[n:n+len(actN),k] = actN
                    n += len(actN)
                
        # Return results
        return self.__constrainedConfiguration(CSB,noDouble,noEmpty)

    # Configuration state for double excitations ===================================
    def __configurationCID(self,reference,active,frozen,noDouble,noEmpty):
        """
        Configuration state basis for single reference double excitations
        """

        # Initialization
        act = np.setdiff1d(active,reference)
        actP = act[act > 0]
        actN = act[act < 0]
        
        frz, _, iFrz = np.intersect1d(frozen,reference,return_indices=True)
        
        nHoN = np.sum(reference < 0)-np.sum(frz < 0)
        nHoP = np.sum(reference > 0)-np.sum(frz > 0)
        nCSB = int(nHoP*(nHoP-1)/2*len(actP)*(len(actP)-1)/2 + nHoN*(nHoN-1)/2*len(actN)*(len(actN)-1)/2 + nHoP*len(actP)*nHoN*len(actN))
        CSB  = np.tile(reference, (nCSB,1))
        
        # Parse through single excitations
        n = 0

        for k in range(len(reference)):
            if all(iFrz != k):
                for l in range(k+1,len(reference)):
                    if all(iFrz != l):
                       if reference[k] > 0:
                           if reference[l] > 0:                                    # both excitations in the up spin channel
                               for m in range(len(actP)):
                                   CSB[n:n+len(actP)-m-1,k] = actP[m]
                                   CSB[n:n+len(actP)-m-1,l] = actP[m+1:]
                                   n += len(actP)-m-1
                           else:                                                   # excitations in different spin channels
                               for m in range(len(actP)):
                                   CSB[n:n+len(actN),k] = actP[m]
                                   CSB[n:n+len(actN),l] = actN
                                   n += len(actN)
                       else:
                           if reference[l] > 0:                                    # excitations in different spin channels
                               for m in range(len(actN)):
                                   CSB[n:n+len(actP),k] = actN[m]
                                   CSB[n:n+len(actP),l] = actP
                                   n += len(actP)
                           else:                                                   # both excitations in the down spin channel
                               for m in range(len(actN)):
                                   CSB[n:n+len(actN)-m-1,k] = actN[m]
                                   CSB[n:n+len(actN)-m-1,l] = actN[m+1:]
                                   n += len(actN)-m-1
                
        # Return results
        return self.__constrainedConfiguration(CSB,noDouble,noEmpty)
            
    # Remove duplicates in the configuration state basis ===========================
    def cleanConfiguration(self):
        """
        Remove duplicate states in the configuration state basis
            Use cleanConfiguration to remove duplicate configuration states in configuration,
            including those related by a permutation of the spin orbitals in the index vector.
            The cleaning process preserves the order of (i) configuration states in the basis,
            keeping only the first unique instance of each state, and (ii) the order of the
            spin-orbital withing each configuration state. For instance, the configuration
                [[-1, -2, 1, 2],[-1, -3, 1, 2],[-2, -1, 1, 2],[-1, -2, 1, 3]]
            is cleaned to
                [[-1, -2, 1, 2],[-1, -3, 1, 2],[-1, -2, 1, 3]]
            since [-2, -1, 1, 2] is a duplicate of [-1, -2, 1, 2], and thus removed from the
            set.
        """

        CSB = np.sort(self.configuration,axis=1);                                   # remove permutations
        CSB, ind = np.unique(CSB,axis=0,return_index=True)                          # identify unique configurations
        self.configuration = self.configuration[np.sort(ind)]                       # clean the configuration basis (keeping the order)

    # Analyze the spectrum of a CI matrix ==========================================
    def analyzeSpectrum(self,CI:np.array,state:np.array=np.empty(0),tolerance:float=1e-2):
        """
        Analyze the spectrum of a CI matrix
            Use analyzeSpectrum to display the spectrum analysis of a CI matrix.

        Parameters:
        -----------
        CI : numpy.array
            CI matrix, typically obtained from computeCI. The CI matrix must be real symmetric.

        state : numpy.array (default numpy.empty(0))
            Indexes of the excited states for which to display the analysis, starting at 1 for
            the first excited state
              * For instance [1, 2, 5] displays the analysis results for the first, second, and
                fifth excited states, skipping the third and fourth.
              * Leave empty to display the analysis for the ground state only

        tolerance : float (default 1e-2)
            Population threshold:
              * For the ground and each of the requested excited states, only configuration
                states that contribute a population larger than tolerance to the wave function
                are listed in the display
        """

        # Initialization
        self.__showHeader()
        print("=== CI matrix spectrum =========================================================")

        # Diagonalize the CI matrix
        if (CI.shape[0] <= 500) or ((len(state) > 0) and (np.max(state) > CI.shape[0]/2)):
            E, V = np.linalg.eigh(CI)                                               # full diagonalization of the CI matrix
        elif len(state) == 0:
            E, V = scp.eigsh(CI,k=1,which='SA')                                     # partial diagonalization of the CI matrix (ground state only)
        else:
            E, V = scp.eigsh(CI,k=np.max(state)+1,which='SA')                       # partial diagonalization of the CI matrix

        # Ground state configuration
        print("  * Ground state")
        print("    Total energy      = %#10.3f a.u. = %#10.3f eV" % (E[0], au2ev(E[0])))
        self.__showComposition(V[:,0],tolerance)

        # Excited states
        for k in state:
            print("\n  * Excited state " + str(k))
            print("    Total energy      = %#10.3f a.u. = %#10.3f eV" % (E[k], au2ev(E[k])))
            print("    Excitation energy = %#10.3f a.u. = %#10.3f eV" % ((E[k]-E[0]), au2ev(E[k]-E[0])))
            self.__showComposition(V[:,k],tolerance)
            

        # Finalization
        self.__showFooter()

    # Display wave function composition ============================================
    def __showComposition(self,V:np.array,tol:float):
        """
        Display the formatted configuration-state make up of a wave function vector
        """
        
        for k in range(len(V)):
            if np.abs(V[k])**2 > tol:
                print("    > %#7.3f %% " % np.abs(10*V[k])**2,end="")
                print(str(self.configuration[k,:]))
