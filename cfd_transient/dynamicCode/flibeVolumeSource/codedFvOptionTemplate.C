/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "codedFvOptionTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "fvMatrix.H"

//{{{ begin codeInclude
#line 110 "/home/patri/projects/hot-flibe-summer/cfd_transient/system/controlDict/functions/C_N16/fvOptions/source_N16"
#include "fvm.H"
//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace fv
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = aefcf76179426655342437a1153d043839abc896
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void flibeVolumeSource_aefcf76179426655342437a1153d043839abc896(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(flibeVolumeSourceFvOptionscalarSource, 0);
addRemovableToRunTimeSelectionTable
(
    option,
    flibeVolumeSourceFvOptionscalarSource,
    dictionary
);

} // End namespace fv
} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fv::
flibeVolumeSourceFvOptionscalarSource::
flibeVolumeSourceFvOptionscalarSource
(
    const word& name,
    const word& modelType,
    const dictionary& dict,
    const fvMesh& mesh
)
:
    fv::cellSetOption(name, modelType, dict, mesh)
{
    if (false)
    {
        printMessage("Construct flibeVolumeSource fvOption from dictionary");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fv::
flibeVolumeSourceFvOptionscalarSource::
~flibeVolumeSourceFvOptionscalarSource()
{
    if (false)
    {
        printMessage("Destroy flibeVolumeSource");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::fv::
flibeVolumeSourceFvOptionscalarSource::correct
(
    GeometricField<scalar, fvPatchField, volMesh>& fld
)
{
    if (false)
    {
        Info<< "flibeVolumeSourceFvOptionscalarSource::correct()\n";
    }

//{{{ begin code
    
//}}} end code
}


void
Foam::fv::
flibeVolumeSourceFvOptionscalarSource::addSup
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "flibeVolumeSourceFvOptionscalarSource::addSup()\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    NotImplemented
//}}} end code
}


void
Foam::fv::
flibeVolumeSourceFvOptionscalarSource::addSup
(
    const volScalarField& rho,
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "flibeVolumeSourceFvOptionscalarSource::addSup(rho)\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 114 "/home/patri/projects/hot-flibe-summer/cfd_transient/system/controlDict/functions/C_N16/fvOptions/source_N16"
// Explicit volumetric source read once from 0/<sourceField>, plus an
                        // implicit decay sink -lambda*rho*psi (phi is a mass flux, so psi is per kg).
                        const word srcName(coeffs_.get<word>("sourceField"));
                        const scalar lambda(coeffs_.get<scalar>("lambda"));
                        if (!mesh_.foundObject<volScalarField>(srcName))
                        {
                            regIOobject::store
                            (
                                new volScalarField
                                (
                                    IOobject
                                    (
                                        srcName, fileName("0"), mesh_,
                                        IOobject::MUST_READ, IOobject::NO_WRITE
                                    ),
                                    mesh_
                                )
                            );
                        }
                        const volScalarField& S = mesh_.lookupObject<volScalarField>(srcName);
                        eqn += S();
                        if (lambda > 0)
                        {
                            eqn -= fvm::Sp(dimensionedScalar(dimless/dimTime, lambda)*rho, eqn.psi());
                        }
//}}} end code
}


void
Foam::fv::
flibeVolumeSourceFvOptionscalarSource::constrain
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "flibeVolumeSourceFvOptionscalarSource::constrain()\n";
    }

//{{{ begin code
    
//}}} end code
}


// ************************************************************************* //

