from __future__ import annotations

import copy
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


class Molecule:
    """
    A wrapper over rdkit molecule to make it easier to use.

    Args:
        mol: rdkit molecule.
        id: an identification of the molecule.
        properties: a dictionary of additional properties associated with the molecule.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        id: Optional[Union[int, str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        self._mol = mol
        self._id = id
        self._properties = properties

        self._charge = None
        self._spin_multiplicity = None
        self._environment = None

    @classmethod
    def from_smiles(cls, s: str, remove_H: bool = True):
        """
        Create a molecule from a smiles string.

        Args:
            s: smiles string, e.g. [CH3+]
            remove_H: whether to remove the H (i.e. move graph H to explicit/implicit
                H). For molecules having atom map number of H, this will remove the atom
                map number as well. In such case, `remove_H = False` will not remove H.
                In either case, other sanitizations are applied to the molecule.
        """
        m = Chem.MolFromSmiles(s, sanitize=remove_H)
        if m is None:
            raise MoleculeError(f"Cannot create molecule for: {s}")

        if not remove_H:
            try:
                Chem.SanitizeMol(m)
            except Exception as e:
                raise MoleculeError(f"Cannot sanitize molecule: {e}")

        return cls(m, s)

    @classmethod
    def from_smarts(cls, s: str, sanitize: bool = True):
        """
        Create a molecule for a smarts string.

        Args:
            s: smarts string, e.g. [Cl:1][CH2:2][CH2:3][CH2:4][C:5](Cl)=[O:6]
            sanitize: whether to sanitize the molecule.
        """
        m = Chem.MolFromSmarts(s)
        if m is None:
            raise MoleculeError(f"Cannot create molecule for: {s}")

        if sanitize:
            Chem.SanitizeMol(m)

        return cls(m, s)

    @classmethod
    def from_sdf(cls, s: str, sanitize: bool = True, remove_H: bool = False):
        """
        Create a molecule for a sdf molecule block.

        We choose to set the default of `remove_H` to `False` because SDF definition of
        explicit and implicit hydrogens is a bit different from what in smiles: it is
        not true that hydrogens specified in SDF are explicit; whether a
        hydrogen is explict or implicit depends on the charge(CHG), valence(VAL) and
        radicals(RAD) specified in the SDF file.

        Args:
            s: SDF mol block string. .
            sanitize: whether to sanitize the molecule
            remove_H: whether to remove hydrogens read from SDF
        """
        m = Chem.MolFromMolBlock(s, sanitize=sanitize, removeHs=remove_H)
        if m is None:
            raise MoleculeError(f"Cannot create molecule for: {s}")
        return cls(m)

    @property
    def rdkit_mol(self) -> Chem.Mol:
        """
        Returns the underlying rdkit molecule..
        """
        return self._mol

    @property
    def id(self) -> Union[int, str, None]:
        """
        Returns the identifier of the molecule.
        """
        return self._id

    @property
    def formal_charge(self) -> int:
        """
        Returns formal charge of the molecule.
        """
        return Chem.GetFormalCharge(self._mol)

    @property
    def charge(self) -> int:
        """
        Returns charge of the molecule.

        The returned charge is the `formal_charge` of the underlying rdkit molecule,
        if charge is set. Otherwise, it is the set charge, which could be different
        from the formal charge.
        """
        if self._charge is None:
            return self.formal_charge
        else:
            return self._charge

    @charge.setter
    def charge(self, charge: int):
        """
        Set the charge of a molecule.

        This will not alter the underlying rdkit molecule at all.

        The charge could be different from the formal charge of the molecule. The purpose
        is to host a charge value that could be used when fragmenting molecules and so on.

        Args:
            charge: charge of the molecule
        """
        self._charge = charge

    @property
    def spin(self) -> int:
        """
        Returns the spin multiplicity of the molecule.

        For example, 0->singlet, 1->doublet, 2->triplet. None-> no spin info available.
        """
        return self._spin_multiplicity

    @spin.setter
    def spin(self, spin: int):
        """
        Set the spin multiplicity for the molecule.
        """
        if spin is not None:
            assert isinstance(spin, int), f"spin should be an integer, got {type(spin)}"
        self._spin_multiplicity = spin

    @property
    def num_atoms(self) -> int:
        """
        Returns number of atoms in molecule
        """
        return self._mol.GetNumAtoms()

    @property
    def num_bonds(self) -> int:
        """
        Returns number of bonds in the molecule.
        """
        return self._mol.GetNumBonds()

    @property
    def bonds(self) -> List[Tuple[int, int]]:
        """
        Returns bond indices specified as tuples of atom indices.
        """
        indices = [
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in self._mol.GetBonds()
        ]

        return indices

    @property
    def species(self) -> List[str]:
        """
        Return species of atoms.
        """
        return [a.GetSymbol() for a in self._mol.GetAtoms()]

    @property
    def composition_dict(self) -> Dict[str, int]:
        """
        Returns composition of the molecule with species as key and number of the species
        as value.
        """
        return Counter(self.species)

    @property
    def formula(self) -> str:
        """
        Returns chemical formula of the molecule, e.g. C1H2O3.
        """
        comp = self.composition_dict
        f = ""
        for s in sorted(comp.keys()):
            f += f"{s}{comp[s]}"

        return f

    @property
    def coords(self) -> np.ndarray:
        """
        Returns coordinates of the atoms. The 2D array is of shape (N, 3), where N is the
        number of atoms.
        """
        self._mol = generate_3D_coords(self._mol)

        # coords = self._mol.GetConformer().GetPositions()
        # NOTE, the above way to get coords results in segfault on linux, so we use the
        # below workaround
        coords = [
            [float(x) for x in self._mol.GetConformer().GetAtomPosition(i)]
            for i in range(self._mol.GetNumAtoms())
        ]

        return np.asarray(coords)

    @property
    def environment(self) -> str:
        """
        Return the computation environment of the molecule, e.g. solvent model.
        """
        return self._environment

    @environment.setter
    def environment(self, value: str):
        """
        Set the computation environment of the molecule, e.g. solvent model.
        """
        self._environment = value

    def set_property(self, name: str, value: Any):
        """
        Add additional property to the molecule.

        If the property is already there this will reset it.

        Args:
            name: name of the property
            value: value of the property
        """
        if self._properties is None:
            self._properties = {}

        self._properties[name] = value

    def get_property(self, name: Union[str, None]):
        """
        Return property of the molecule.

        If name is `None`, return a dictionary of all properties.

        Args:
            name: property name
        """
        if self._properties is None:
            raise MoleculeError("Molecule does not have any additional property")
        else:
            if name is None:
                return self._properties
            else:
                try:
                    return self._properties[name]
                except KeyError:
                    raise MoleculeError(f"Molecule does not have property {name}")

    def get_atom_map_number(self) -> List[Union[int, None]]:
        """
        Get the atom map number in the rdkit molecule.

        Returns:
            Atom map number for each atom. Index in the returned list is the atom index.
            If an atom is not mapped, the map number is set to `None`.
        """

        map_number = []
        for i, atom in enumerate(self._mol.GetAtoms()):
            if atom.HasProp("molAtomMapNumber"):
                map_number.append(atom.GetAtomMapNum())
            else:
                map_number.append(None)

        return map_number

    def set_atom_map_number(self, map_number: Dict[int, int]):
        """
        Set the atom map number of the rdkit molecule.

        Args:
            Atom map number for each atom. If a value is `None`, the atom map number
            in the rdkit molecule is cleared.
        """
        for idx, number in map_number.items():
            if idx >= self.num_atoms:
                raise MoleculeError(
                    f"Cannot set atom map number of atom {idx} (starting from 0) for "
                    f"a molecule has a total number of {self.num_atoms} atoms."
                )

            atom = self._mol.GetAtomWithIdx(idx)

            if number is None:
                atom.ClearProp("molAtomMapNumber")
            elif number <= 0:
                raise MoleculeError(
                    f"Expect atom map number larger than 0, but got  {number}."
                )
            else:
                atom.SetAtomMapNum(number)

    def clear_atom_map_number(self):
        """
        Remove the atom map number for all atoms.
        """
        map_number = {i: None for i in range(self.num_atoms)}
        self.set_atom_map_number(map_number)

        return self

    def generate_coords(self) -> np.ndarray:
        """
        Generate 3D coordinates for an rdkit molecule by embedding it.

        This only generates coords, but the coords are not optimized.

        Returns:
            A 2D array of shape (N, 3), where N is the number of atoms.
        """
        error = AllChem.EmbedMolecule(self._mol, randomSeed=35)
        if error == -1:  # https://sourceforge.net/p/rdkit/mailman/message/33386856/
            AllChem.EmbedMolecule(self._mol, randomSeed=35, useRandomCoords=True)
        if error == -1:
            raise MoleculeError(
                "Cannot generate coordinates for molecule; embedding fails."
            )

        return self.coords

    def optimize_coords(self) -> np.ndarray:
        """
        Optimize atom coordinates using MMFF and UFF force fields.

        Returns:
            optimized coords, a 2D array of shape (N, 3), where N is the number of atoms.
        """

        # TODO usually, you need to add the H
        def optimize_till_converge(method, m):
            maxiters = 200
            while True:
                error = method(m, maxIters=maxiters)
                if error == 1:
                    maxiters *= 2
                else:
                    return error

        # generate conformer if not exists
        try:
            self._mol.GetConformer()
        except ValueError:
            self.generate_coords()

        # optimize, try MMFF first, if fails then UFF
        error = optimize_till_converge(AllChem.MMFFOptimizeMolecule, self._mol)
        if error == -1:  # MMFF cannot be set up
            optimize_till_converge(AllChem.UFFOptimizeMolecule, self._mol)

        return self.coords

    def to_smiles(self) -> str:
        """
        Returns a smiles representation of the molecule.
        """
        return Chem.MolToSmiles(self._mol)

    def to_sdf(
        self,
        filename: Optional[Union[Path]] = None,
        kekulize: bool = True,
        v3000: bool = True,
        name: Optional[str] = None,
    ) -> str:
        """
        Convert molecule to an sdf representation.

        Args:
            filename: if not None, write to the path.
            kekulize: whether to kekulize the mol
            v3000: if `True` write in SDF v3000 format, otherwise, v2000 format.
            name: Name of the molecule, i.e. first line of the sdf file. If None,
            molecule.id will be used.

        Returns:
             a sdf representation of the molecule.
        """
        name = str(self._id) if name is None else name
        self._mol.SetProp("_Name", name)

        sdf = Chem.MolToMolBlock(self._mol, kekulize=kekulize, forceV3000=v3000)
        if filename is not None:
            with open(filename, "w") as f:
                f.write(sdf)

        return sdf

    def draw(
        self, filename: Optional[Union[Path]] = None, with_atom_index: bool = False
    ) -> Chem.Mol:
        """
        Draw the molecule.

        Args:
            filename: path to the save the generated image. If `None`,
                image will not be generated.
            with_atom_index: whether to show the atom index in the image.

        Returns:
            the molecule (which will then show up in Jupyter notebook)
        """
        # compute better coords to show in 2D
        m = copy.deepcopy(self._mol)
        AllChem.Compute2DCoords(m)

        if with_atom_index:
            for a in m.GetAtoms():
                a.SetAtomMapNum(a.GetIdx() + 1)

        if filename is not None:
            Draw.MolToFile(m, str(filename))

        return m

    def draw_with_bond_note(
        self,
        note: Dict[Tuple[int, int], str],
        filename: Optional[Union[Path]] = None,
        with_atom_index: bool = False,
        image_size=(400, 300),
        format: bool = "svg",
    ):
        """
        Draw molecule and show a note along bond.

        The returned image can be viewed in Jupyter with display(SVG(image)).

        Args:
            note: {bond_index: note}. The note to show for the corresponding bond.
                `bond index` is a tuple of atom indices.
            filename: path to the save the generated image. If `None`,
                image will not be generated, but instead, will show in Jupyter notebook.
            with_atom_index: whether to show the atom index in the image.
            format: format of the image, `png` or `svg`
        """
        m = self.draw(with_atom_index=with_atom_index)

        # set bond annotation
        highlight_bonds = []
        for bond, note in note.items():
            if isinstance(note, (float, np.floating)):
                note = "{:.3g}".format(note)
            idx = m.GetBondBetweenAtoms(*bond).GetIdx()
            m.GetBondWithIdx(idx).SetProp("bondNote", note)
            highlight_bonds.append(idx)

        # set highlight color
        bond_colors = {b: (192 / 255, 192 / 255, 192 / 255) for b in highlight_bonds}

        if format == "png":
            d = rdMolDraw2D.MolDraw2DCairo(*image_size)
        elif format == "svg":
            d = rdMolDraw2D.MolDraw2DSVG(*image_size)
        else:
            supported = ["png", "svg"]
            raise ValueError(f"Supported format are {supported}; got {format}")

        # smaller font size
        d.SetFontSize(0.8 * d.FontSize())

        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, highlightBonds=highlight_bonds, highlightBondColors=bond_colors
        )
        d.FinishDrawing()
        img = d.GetDrawingText()

        if filename is not None:
            with open(filename, "wb") as f:
                f.write(img)

        return img

    def sanitize(self):
        """
        Sanitize the molecule.
        """
        Chem.SanitizeMol(self._mol)

        return self

    def add_H(self, explicit_only: bool = False) -> Molecule:
        """
        Add hydrogens to the molecule.

        Args:
            explicit_only: only add explicit hydrogens to the graph

        Returns:
            The molecule with hydrogens added.
        """
        self._mol = Chem.AddHs(self._mol, explicitOnly=explicit_only)
        self.sanitize()

        return self

    def remove_H(self, implicit_only: bool = False) -> Molecule:
        """
        Remove hydrogens to the molecule.

        Args:
            implicit_only: only remove implicit hydrogens from the graph

        Returns:
            The molecule with hydrogens removed.
        """
        self._mol = Chem.RemoveHs(self._mol, implicitOnly=implicit_only)
        self.sanitize()

        return self


def generate_3D_coords(m: Chem.Mol) -> Chem.Mol:
    """
    Generate 3D coords for an rdkit molecule.

    This is done by embedding and then optimizing using MMFF force filed (or UFF force
    field).

    Args:
        m: rdkit mol.

    Returns:
        rdkit mol with updated coords
    """

    def optimize_till_converge(method, m):
        maxiters = 200
        while True:
            error = method(m, maxIters=maxiters)
            if error == 1:
                maxiters *= 2
            else:
                return error

    # embedding
    error = AllChem.EmbedMolecule(m, randomSeed=35)
    if error == -1:  # https://sourceforge.net/p/rdkit/mailman/message/33386856/
        AllChem.EmbedMolecule(m, randomSeed=35, useRandomCoords=True)
    if error == -1:
        raise MoleculeError("Cannot generate coords for mol.")

    # optimize, try MMFF first, if fails then UFF
    error = optimize_till_converge(AllChem.MMFFOptimizeMolecule, m)
    if error == -1:  # MMFF cannot be set up
        optimize_till_converge(AllChem.UFFOptimizeMolecule, m)

    return m


def find_functional_group(
    mol: Chem.Mol, atoms: List[int], func_groups: Union[Path, List]
) -> List[int]:
    """
    Find the largest functional group associated with the give atoms.

    This will loop over all the given functional groups, check whether a functional
    group contains (some of or all) the given atoms. If yes, then
    1. if the number of given atoms it contains is larger than the present one,
    this functional group is selected.
    2. if the number of given atoms it contains is the same but the number of atoms in
    the functional group is larger than the already selected one, this new functional
    group is selected.


    Args:
        mol: rdkit mol
        atoms: a list of atoms index (index of rdkit atoms, not map number) starting
            from 0.
        func_groups: if a Path, should be a Path to a tsv file containing the SMARTS
            of the functional group. Or it could be a list of rdkit mols
            created by MolFromSmarts.

    Returns:
        functional group, specified by a list of atom indexes. Note the functional
            group may not include all the given atoms.
    """

    fg_atoms = []  # atom index of functional group
    num_in_fg = 0  # number of given atoms in functional group

    atoms = set(atoms)

    for fg in func_groups:
        sub = mol.GetSubstructMatch(fg)
        intersect = atoms.intersection(sub)

        if intersect:
            if len(intersect) > num_in_fg:
                fg_atoms = sub
                num_in_fg = len(intersect)

            elif len(intersect) == num_in_fg:
                if len(sub) > len(fg_atoms):
                    fg_atoms = sub

    return fg_atoms


class MoleculeError(Exception):
    def __init__(self, msg=None):
        super(MoleculeError, self).__init__(msg)
        self.msg = msg
