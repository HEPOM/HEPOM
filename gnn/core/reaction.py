import itertools
import copy
import logging
from collections.abc import Iterable
import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
from pymatgen.analysis.graphs import _isomorphic
from collections import defaultdict, OrderedDict
from gnn.core.molwrapper import rdkit_mol_to_wrapper_mol
from gnn.core.rdmol import fragment_rdkit_mol
from gnn.utils import pickle_dump

logger = logging.getLogger(__name__)


class Reaction:
    """
    A reaction that only has one bond break or the type ``A -> B + C``
    (break a bond not in a ring) or ``A -> B`` (break a bond in a ring).

    Args:
        reactants (list): MoleculeWrapper instances
        products (list): MoleculeWrapper instances
        broken_bond (tuple or None): indices of atoms associated with the broken bond
        free_energy (float or None): free energy of the reaction
        identifier (str): (unique) identifier of the reaction

    Note:
        most methods in this class only works for A->B and A->B+C type reactions
    """

    def __init__(
        self, reactants, products, broken_bond=None, free_energy=None, identifier=None
    ):

        assert len(reactants) == 1, "incorrect number of reactants, should be 1"
        assert 1 <= len(products) <= 2, "incorrect number of products, should be 1 or 2"

        self.reactants = reactants
        self.products = products
        self._broken_bond = broken_bond
        self._free_energy = free_energy
        self._id = identifier

        self._atom_mapping = None
        self._bond_mapping_by_int_index = None
        self._bond_mapping_by_tuple_index = None
        self._bond_mapping_by_sdf_int_index = None

    def get_id(self):
        if self._id is None:
            # set id to reactant id and broken bond of reactant
            mol = self.reactants[0]
            broken_bond = "-".join([str(i) for i in self.get_broken_bond()])
            species = "-".join(sorted(self.get_broken_bond_attr()["species"]))
            str_id = str(mol.id) + "_broken_bond-" + broken_bond + "_species-" + species
            self._id = str_id

        return self._id

    def get_free_energy(self):
        if self._free_energy is not None:
            return self._free_energy
        else:
            energy = 0.0
            for mol in self.reactants:
                if mol.free_energy is None:
                    return None
                else:
                    energy -= mol.free_energy
            for mol in self.products:
                if mol.free_energy is None:
                    return None
                else:
                    energy += mol.free_energy
            return energy

    def get_broken_bond(self):
        """
        Returns:
            tuple: sorted index of broken bond (a 2-tuple of atom index)
        """
        if self._broken_bond is None:
            if len(self.products) == 1:
                bonds = is_valid_A_to_B_reaction(
                    self.reactants[0], self.products[0], first_only=True
                )
            else:
                bonds = is_valid_A_to_B_C_reaction(
                    self.reactants[0], self.products[0], self.products[1], first_only=True
                )
            if not bonds:
                msg = f"Reaction id: {self.get_id()}. "
                msg += "Reactants id: "
                for m in self.reactants:
                    msg += f"{m.id} "
                msg += "Products id: "
                for m in self.products:
                    msg += f"{m.id} "
                raise RuntimeError(
                    f"Cannot break a reactant bond to get products; "
                    f"invalid reaction: {msg}"
                )
            # only one element in `bonds` because of `first_only = True`
            self._broken_bond = bonds[0]

        return tuple(sorted(self._broken_bond))

    def get_broken_bond_attr(self):
        """
        Returns a dict of attributes of the broken bond.
        """
        reactant = self.reactants[0]
        u, v = self.get_broken_bond()
        species = (reactant.species[u], reactant.species[v])

        return {"species": species}

    def atom_mapping(self):
        """
        Find the atom mapping between products and reactant.

        For example, suppose we have reactant

              C 0
             / \
            /___\
           O     N---H
           1     2   3

        products
               C 0
             / \
            /___\
           O     N
           1     2
        and (note the index of H changes it 0)
           H 0

        This function generates the atom mapping:
        [{0:0, 1:1, 2:2}, {0:3}]
        where the first dict is the mapping for product 1 and the second dict is the
        mapping for product 2.

        Returns:
            list: each element is a dict that maps the atoms from product to reactant.
        """
        if self._atom_mapping is not None:
            return self._atom_mapping

        # get subgraphs of reactant by breaking the bond
        # if A->B reaction, there is one element in subgraphs
        # if A->B+C reaction, there are two
        bond = self.get_broken_bond()
        original = copy.deepcopy(self.reactants[0].mol_graph)
        original.break_edge(bond[0], bond[1], allow_reverse=True)
        components = nx.weakly_connected_components(original.graph)
        subgraphs = [original.graph.subgraph(c) for c in components]

        # correspondence between products and reactant subgrpahs
        if len(subgraphs) == 1:
            corr = {0: 0}
        else:
            # product idx as key and reactant subgraph idx as value
            # order matters since mappings[0] (see below) corresponds to first product
            corr = OrderedDict()
            products = [p.mol_graph for p in self.products]

            # implicitly indicating _isomorphic(subgraphs[1], products[1].graph)
            if _isomorphic(subgraphs[0], products[0].graph):
                corr[0] = 0
                corr[1] = 1
            else:
                corr[0] = 1
                corr[1] = 0

        # atom mapping between products and reactant
        mappings = []
        for pidx, fidx in corr.items():
            mp = nx_graph_atom_mapping(self.products[pidx].graph, subgraphs[fidx])
            if mp is None:
                raise RuntimeError(f"cannot find atom mapping for reaction {str(self)}")
            else:
                mappings.append(mp)

        self._atom_mapping = mappings

        return self._atom_mapping

    def bond_mapping_by_int_index(self):
        r"""
        Find the bond mapping between products and reactant, using a single index (the
        index of bond in MoleculeWrapper.bonds) to denote the bond.

        For example, suppose we have reactant

              C 0
           0 / \ 1
            /___\  3   4
           O  2  N---O---H
           1     2   3  4

        products
              C 0
           1 / \ 0
            /___\
           O  2  N
           1     2
        and (note the index of H changes it 0)
              0
            O---H
            0   1
        The function gives the bond mapping:
        [{0:1, 1:0, 2:2}, {0:4}]


        The mapping is done by finding correspondence between atoms indices of reactant
        and products.

        Returns:
            list: each element is a dict mapping the bonds from product to reactant
        """

        if self._bond_mapping_by_int_index is not None:
            return self._bond_mapping_by_int_index

        # mapping between tuple index and integer index for the same bond
        reactant_mapping = {
            bond: ordering
            for ordering, (bond, _) in enumerate(self.reactants[0].bonds.items())
        }

        atom_mapping = self.atom_mapping()
        bond_mapping = []

        for p, amp in zip(self.products, atom_mapping):
            bmp = dict()

            for p_ordering, (bond, _) in enumerate(p.bonds.items()):

                # atom mapping between product and reactant of the bond
                bond_amp = [amp[i] for i in bond]

                r_ordering = reactant_mapping[tuple(sorted(bond_amp))]
                bmp[p_ordering] = r_ordering
            bond_mapping.append(bmp)

        self._bond_mapping_by_int_index = bond_mapping

        return self._bond_mapping_by_int_index

    def bond_mapping_by_tuple_index(self):
        r"""
        Find the bond mapping between products and reactant, using a tuple index (atom
        index) to denote the bond.

        For example, suppose we have reactant

              C 0
           0 / \ 1
            /___\  3   4
           O  2  N---O---H
           1     2   3  4

        products
              C 0
           1 / \ 0
            /___\
           O  2  N
           2     1
        and (note the index of H changes it 0)
              0
            O---H
            0   1
        The function will give the bond mapping:
        [{(0,1):(0,2), (0,2):(0,1), (1,2):(1,2)}, {(0,1):(3,4)}]


        The mapping is done by finding correspondence between atoms indices of reactant
        and products.

        Returns:
            list: each element is a dict mapping the bonds from product to reactant
        """

        if self._bond_mapping_by_tuple_index is not None:
            return self._bond_mapping_by_tuple_index

        atom_mapping = self.atom_mapping()
        bond_mapping = []

        for p, amp in zip(self.products, atom_mapping):
            bmp = dict()

            for b_product, _ in p.bonds.items():

                # atom mapping between product and reactant of the bond
                i, j = b_product

                b_reactant = tuple(sorted([amp[i], amp[j]]))
                bmp[b_product] = b_reactant
            bond_mapping.append(bmp)

        self._bond_mapping_by_tuple_index = bond_mapping

        return self._bond_mapping_by_tuple_index

    def bond_mapping_by_sdf_int_index(self):
        """
        Bond mapping between products SDF bonds (integer index) and reactant SDF bonds
        (integer index).

        Unlike the atom mapping (where atom index in graph and sdf are the same),
        the ordering of bond may change when sdf file are written. So we need this
        mapping to ensure the correct ordering between products bonds and reactant bonds.

        We do the below to get a mapping between product sdf int index and reactant
        sdf int index:

        product sdf int index
        --> product sdf tuple index
        --> product graph tuple index
        --> reactant graph tuple index
        --> reactant sdf tuple index
        --> reactant sdf int index


        Returns:
            list (dict): each dict is the mapping for one product, from sdf bond index
                of product to sdf bond index of reactant
        """

        if self._bond_mapping_by_sdf_int_index is not None:
            return self._bond_mapping_by_sdf_int_index

        reactant = self.reactants[0]

        # reactant sdf bond index (tuple) to sdf bond index (integer)
        reactant_index_tuple2int = {
            b: i for i, b in enumerate(reactant.get_sdf_bond_indices(zero_based=True))
        }

        # bond mapping between product sdf and reactant sdf
        bond_mapping = []
        product_to_reactant_mapping = self.bond_mapping_by_tuple_index()
        for p, p2r in zip(self.products, product_to_reactant_mapping):

            mp = {}
            # product sdf bond index (list of tuple)
            psb = p.get_sdf_bond_indices(zero_based=True)

            # ib: product sdf bond index (int)
            # b: product graph bond index (tuple)
            for ib, b in enumerate(psb):

                # reactant graph bond index (tuple)
                rsbt = p2r[b]

                # reactant sdf bond index (int)
                rsbi = reactant_index_tuple2int[rsbt]

                # product sdf bond index (int) to reactant sdf bond index (int)
                mp[ib] = rsbi

            # list of dict, each dict for one product
            bond_mapping.append(mp)

        self._bond_mapping_by_sdf_int_index = bond_mapping
        return self._bond_mapping_by_sdf_int_index

    def as_dict(self):
        d = {
            "reactants": [
                "{}_{}_{}_{}".format(m.formula, m.charge, m.id, m.free_energy)
                for m in self.reactants
            ],
            "products": [
                "{}_{}_{}_{}".format(m.formula, m.charge, m.id, m.free_energy)
                for m in self.products
            ],
            "charge": [m.charge for m in self.reactants + self.products],
            "broken_bond": self.get_broken_bond(),
            "bond_energy": self.get_free_energy(),
        }
        return d

    def __expr__(self):
        if len(self.products) == 1:
            s = "\nA -> B style reaction\n"
        else:
            s = "\nA -> B + C style reaction\n"
        s += "reactants:\n"
        for p in self.reactants:
            s += f"    {p.formula} ({p.charge})\n"
        s += "products:\n"
        for p in self.products:
            s += f"    {p.formula} ({p.charge})\n"

        return s

    def __str__(self):
        return self.__expr__()

    def __eq__(self, other):
        # this assumes molecule id is unique
        self_ids = {m.id for m in self.reactants + self.products}
        other_ids = {m.id for m in other.reactants + other.products}

        return self_ids == other_ids


class ReactionsGroup:
    """
    A group of reactions that have the same reactant.

    This is a base class, so use the derived class.
    """

    def __init__(self, reactant, reactions=None):
        self._reactant = reactant
        self._reactions = []
        if reactions is not None:
            self.add(reactions)

    @property
    def reactant(self):
        """Return the reactant, which is the same for all reactions."""
        return self._reactant

    @property
    def reactions(self):
        """Return a list of Reaction."""
        return self._reactions

    def add(self, reactions):
        """
        Add one or more reactions to the collection.

        Args:
            reactions: A `Reaction` or a sequence of `Reaction`.

        """
        if isinstance(reactions, Iterable):
            for rxn in reactions:
                self._add_one(rxn)
        else:
            self._add_one(reactions)

    def _add_one(self, rxn):
        if rxn.reactants[0] != self.reactant:
            raise ValueError(
                "Cannot add reaction whose reactant is different from what already "
                "in the collection."
            )
        self._reactions.append(rxn)


class ReactionsOfSameBond(ReactionsGroup):
    """
    A collection of reactions associated with the same bond in the reactant.

    This is mainly to consider products with the same mol graph (i.e. isomorphic to
    each other) but different charges.
    """

    def __init__(self, reactant, reactions=None, broken_bond=None):
        super(ReactionsOfSameBond, self).__init__(reactant, reactions)
        self._broken_bond = broken_bond

    @property
    def broken_bond(self):
        if self._broken_bond is None:
            if self._reactions:
                self._broken_bond = self.reactions[0].get_broken_bond()
            else:
                raise RuntimeError(
                    "Cannot get broken bond. You can either provide it at instantiation "
                    "or add some reactions and then try again."
                )
        return self._broken_bond

    def create_complement_reactions(self, allowed_charge=[-1, 0, 1], mol_reservoir=None):
        """
        Create reactions to complement the ones present in the database such that each
        bond has reactions of all combination of charges.

        For example, if we have `A (0) -> B (0) + C (0)` and `A (0) -> B (1) + C (-1)`
        in the database, this will create reaction `A (0) -> B (-1) + C (1)`,
        assuming molecule charges {-1,0,1} are allowed.

        Args:
            allowed_charge (list): allowed charges for molecules (products).
            mol_reservoir (set): For newly created complement reactions, a product
                is first searched in the mol_reservoir. If existing (w.r.t. charge and
                isomorphism), the mol from the reservoir is used as the product; if
                not, new mol is created. Note, if a mol is not in `mol_reservoir`,
                it is added to comp_mols.

        Returns:
            comp_rxns (list): A sequence of `Reaction`s that complement the existing ones.
            comp_mols (set): new molecules created to setup the `comp_rxns`.
        """

        # find products charges
        fragments = self.reactant.fragments[self.broken_bond]
        N = len(fragments)

        # A -> B reaction
        if N == 1:
            # For N = 1 case, there could only be one reaction where the charges of
            # the reactant and product are the same. If we already have one reaction,
            # no need to find the complementary one.
            if len(self.reactions) == 1:
                return [], set()
            else:
                missing_charge = [[self.reactant.charge]]

        # N == 2 case, i.e. A -> B + C reaction (B could be the same as C)
        else:
            target_products_charge = factor_integer(
                self.reactant.charge, allowed_charge, num=N
            )

            # all possible reactions are present
            if len(target_products_charge) == len(self.reactions):
                return [], set()
            else:
                products_charge = []
                for rxn in self.reactions:
                    products = [p.mol_graph for p in rxn.products]
                    charge = [p.charge for p in rxn.products]

                    # A->B+B reaction
                    if products[0].isomorphic_to(products[1]):
                        products_charge.append((charge[0], charge[1]))
                        products_charge.append((charge[1], charge[0]))

                    # A->B+C reaction
                    else:
                        # implicitly indicates fragments[1].isomorphic_to(products[1])
                        if fragments[0].isomorphic_to(products[0]):
                            products_charge.append((charge[0], charge[1]))
                        else:
                            products_charge.append((charge[1], charge[0]))

                missing_charge = list(set(target_products_charge) - set(products_charge))

        # create reactions and mols
        comp_rxns, comp_mols = create_reactions_from_reactant(
            self.reactant, self.broken_bond, missing_charge, mol_reservoir=mol_reservoir,
        )

        return comp_rxns, comp_mols

    def order_reactions(self, complement_reactions=False, mol_reservoir=None):
        """
        Order reactions by energy.

        If complement reactions (whose energy is `None`) are used, they are placed
        after reactions having energies.

        Args:
            complement_reactions (bool): If `False`, order the existing reactions only.
                Otherwise, complementary reactions are created and ordered together
                with existing ones.
            mol_reservoir (set): If `complement_reactions` is False, this is silently
                ignored. Otherwise, for newly created complement reactions, a product
                is first searched in the mol_reservoir. If existing (w.r.t. charge and
                isomorphism), the mol from the reservoir is used as the product; if
                not, new mol is created. Note, if a mol is not in `mol_reservoir`,
                it will be added to `mol_reservoir`, i.e. `mol_reservoir` is updated
                inplace.

        Returns:
            list: a sequence of :class:`Reaction` ordered by energy
        """
        ordered_rxns = sorted(self.reactions, key=lambda rxn: rxn.get_free_energy())
        if complement_reactions:
            comp_rxns, comp_mols = self.create_complement_reactions(
                mol_reservoir=mol_reservoir
            )
            ordered_rxns += comp_rxns
            if mol_reservoir is not None:
                mol_reservoir.update(comp_mols)

        return ordered_rxns


class ReactionsMultiplePerBond(ReactionsGroup):
    """
    A collection of reactions for the same reactant.

    Each bond can be associated with multiple reactions of different charges.
    """

    def group_by_bond(self, find_one=True):
        """
        Group reactions with same broken bond together.

        If there is not reactions associated with a bond of the reactant, the
        corresponding :class:`ReactionsOfSameBond` is still created, but initialized
        with empty reactions.

        Args:
            find_one (bool): If `True`, keep one reaction for each isomorphic bond
                group. If `False`, keep all.
                Note, if set to `True`, this expects that `find_one=False` in
                :method:`ReactionExtractorFromMolSet..extract_one_bond_break` so that all bonds
                in an isomorphic group have exactly the same reactions.
                In such, we just need to retain a random bond and its associated
                reactions in each group.

        Returns:
            list: a sequence of :class:`ReactionsOfSameBond`, one for each bond of the
            reactant.
        """

        # init an empty [] for each bond
        # doing this instead of looping over self.reactions ensures bonds without
        # reactions are correctly represented
        bond_rxns_dict = {b: [] for b, _ in self.reactant.bonds.items()}

        # assign rxn to bond group
        for rxn in self.reactions:
            bond_rxns_dict[rxn.get_broken_bond()].append(rxn)

        # remove duplicate isomorphic bonds
        if find_one:
            for group in self.reactant.isomorphic_bonds:
                # keep the first bond in each group and remove others
                for i in range(1, len(group)):
                    bond_rxns_dict.pop(group[i])

        # create ReactionsOfSameBond instance
        reactions = []
        for bond, rxns in bond_rxns_dict.items():
            rsb = ReactionsOfSameBond(self.reactant, reactions=rxns, broken_bond=bond)
            reactions.append(rsb)

        return reactions

    def order_reactions(
        self, one_per_iso_bond_group=True, complement_reactions=False, mol_reservoir=None
    ):
        """
        Order reactions by energy.

        If complement reactions (whose energy is `None`) are used, they are placed
        after reactions having energies.

        Args:
            one_per_iso_bond_group (bool): If `True`, keep one reaction for each
                isomorphic bond group. If `False`, keep all.
                Note, if set to `True`, this expects that `find_one=False` in
                :method:`ReactionExtractorFromMolSet.extract_one_bond_break` so that all bonds
                in an isomorphic bond group have exactly the same reactions.
            complement_reactions (bool): If `False`, order the existing reactions only.
                Otherwise, complementary reactions are created and ordered together
                with existing ones. The complementary reactions are ordered later than
                existing reactions.
            mol_reservoir (set): If `complement_reactions` is False, this is silently
                ignored. Otherwise, for newly created complement reactions, a product
                is first searched in the mol_reservoir. If existing (w.r.t. charge and
                isomorphism), the mol from the reservoir is used as the product; if
                not, new mol is created.

        Note:
            If a mol is not in `mol_reservoir`, it will be added to `mol_reservoir`,
            i.e. `mol_reservoir` is updated inplace.

        Returns:
            list: a sequence of :class:`Reaction` ordered by energy.
        """

        # NOTE, we need to get existing_rxns from rsb instead of self.reactions because
        # we may need to remove duplicate isomorphic-bond rxn, which is handled in
        # self.group_by_bonds.

        existing_rxns = []
        rsb_group = self.group_by_bond(find_one=one_per_iso_bond_group)
        for rsb in rsb_group:
            existing_rxns.extend(rsb.reactions)

        # sort reactions we have energy for
        ordered_rxns = sorted(existing_rxns, key=lambda rxn: rxn.get_free_energy())

        if complement_reactions:
            for rsb in rsb_group:
                comp_rxns, comp_mols = rsb.create_complement_reactions(
                    allowed_charge=[-1, 0, 1], mol_reservoir=mol_reservoir
                )
                ordered_rxns += comp_rxns
                if mol_reservoir is not None:
                    mol_reservoir.update(comp_mols)

        return ordered_rxns


class ReactionsOnePerBond(ReactionsMultiplePerBond):
    """
    A collection of reactions for the same reactant.
    There is at most one reaction associated with a bond, either a specific choice
    of charges (e.g. 0 -> 0 + 0) or lowest energy reaction across charges.
    """

    def _add_one(self, rxn):
        if rxn.reactants[0] != self.reactant:
            raise ValueError(
                "Cannot add reaction whose reactant is different from what already in "
                "the collection."
            )
        bond = rxn.get_broken_bond()
        for r in self.reactions:
            if r.get_broken_bond() == bond:
                raise ValueError(
                    f"Reaction breaking bond {bond} already exists.\n"
                    f"Existing reaction: {str(r.as_dict())}\n"
                    f"New      reaction: {str(rxn.as_dict())}"
                )
        self._reactions.append(rxn)

    def order_reactions(
        self, one_per_iso_bond_group=True, complement_reactions=False, mol_reservoir=None
    ):
        """
        Order reactions by energy.

        If complement reactions (whose energy is `None`) are used, they are placed
        after reactions having energies.

        Args:
            complement_reactions (bool): If `False`, order the existing reactions only.
                Otherwise, complementary reactions are created and ordered together
                with existing ones.
            one_per_iso_bond_group (bool): If `True`, keep one reaction for each
                isomorphic bond group. If `False`, keep all.
                Note, if set to `True`, this expects that `find_one=False` in
                :method:`ReactionExtractorFromMolSet.extract_one_bond_break` so that all bonds
                in an isomorphic bond group have exactly the same reactions.
            mol_reservoir (set): If `complement_reactions` is False, this is silently
                ignored. Otherwise, for newly created complement reactions, a product
                is first searched in the mol_reservoir. If existing (w.r.t. charge and
                isomorphism), the mol from the reservoir is used as the product; if
                not, new mol is created.

        Note:
            If a mol is not in `mol_reservoir`, it will be added to `mol_reservoir`,
            i.e. `mol_reservoir` is updated inplace.

        Returns:
            list: a sequence of :class:`Reaction` ordered by energy.
        """

        # NOTE, we need to get existing_rxns from rsb instead of self.reactions because
        # we may need to remove duplicate isomorphic-bond rxn, which is handled in
        # self.group_by_bonds.

        existing_rxns = []
        rsb_group = self.group_by_bond(find_one=one_per_iso_bond_group)
        for rsb in rsb_group:
            existing_rxns.extend(rsb.reactions)

        # sort reactions we have energy for
        ordered_rxns = sorted(existing_rxns, key=lambda rxn: rxn.get_free_energy())

        if complement_reactions:
            for rsb in rsb_group:
                comp_rxns, comp_mols = rsb.create_complement_reactions(
                    allowed_charge=[0], mol_reservoir=mol_reservoir
                )
                ordered_rxns += comp_rxns
                if mol_reservoir is not None:
                    mol_reservoir.update(comp_mols)

        return ordered_rxns


class ReactionExtractorFromMolSet:
    """
    Compose reactions from a set of molecules.

    This currently supports two types of (one-bond-break) reactions:
    A -> B
    A -> B + C

    A reaction is determined to be valid if:
    - balance of mass
    - balance of charge
    - connectivity: molecule graph between reactant and products only differ by one
        edge (bond)

    Args:
        molecules (list): a sequence of :class:`MoleculeWrapper` molecules.
    """

    def __init__(self, molecules):
        self.molecules = molecules
        self.reactions = None

    def bucket_molecules(self, keys=["formula", "charge"]):
        """
        Classify molecules into nested dictionaries according to molecule properties
        specified in ``keys``.

        Args:
            keys (list of str): each str should be a molecule property.

        Returns:
            nested dictionary of molecules classified according to keys.
        """
        logger.info("Start bucketing molecules...")

        num_keys = len(keys)
        buckets = {}
        for m in self.molecules:
            b = buckets
            for i, k in enumerate(keys):
                v = getattr(m, k)
                if i == num_keys - 1:
                    b.setdefault(v, []).append(m)
                else:
                    b.setdefault(v, {})
                b = b[v]

        return buckets

    def extract_A_to_B_style_reaction(self, find_one=True):
        """
        Extract a list of A -> B reactions.

        Args:
            find_one (bool): For a given reactant A and product B, there could be
                multiple reactions between them (breaking different bonds of reactant
                results in the same product). If `True`, for each A and B only get
                one reaction between them. If `False` get all.

        Returns:
            list: a sequence of :class:`Reaction`.
        """

        logger.info("Start extracting A -> B style reactions")

        buckets = self.bucket_molecules(keys=["formula", "charge"])

        A2B = []
        i = 0
        for formula in buckets:
            i += 1
            if i % 10000 == 0:
                logger.info(f"A -> B running bucket {i}")

            for charge in buckets[formula]:
                for A, B in itertools.permutations(buckets[formula][charge], 2):
                    bonds = is_valid_A_to_B_reaction(A, B, first_only=find_one)
                    for b in bonds:
                        A2B.append(Reaction([A], [B], b))

        self.reactions = A2B

        logger.info("{} A -> B style reactions extracted".format(len(A2B)))

        return A2B

    def extract_A_to_B_C_style_reaction(self, find_one=True):
        """
        Extract a list of A -> B + C reactions (B and C can be the same molecule).

        Args:
            find_one (bool): For a given reactant A and product B, there could be
                multiple reactions between them (breaking different bonds of reactant
                results in the same product). If `True`, for each A and B only get
                one reaction between them. If `False` get all.

        Returns:
            list: a sequence of :class:`Reaction`.
        """

        logger.info("Start extracting A -> B + C style reactions")

        buckets = self.bucket_molecules(keys=["formula", "charge"])

        fcmap = self._get_formula_composition_map(self.molecules)

        A2BC = []
        i = 0
        for formula_A in buckets:
            for formula_B, formula_C in itertools.combinations_with_replacement(
                buckets, 2
            ):
                i += 1
                if i % 10000 == 0:
                    logger.info(f"A -> B + C running bucket {i}")

                if not self._is_valid_A_to_B_C_composition(
                    fcmap[formula_A], fcmap[formula_B], fcmap[formula_C]
                ):
                    continue

                reaction_ids = []
                for (charge_A, charge_B, charge_C) in itertools.product(
                    buckets[formula_A], buckets[formula_B], buckets[formula_C]
                ):
                    if not self._is_valid_A_to_B_C_charge(charge_A, charge_B, charge_C):
                        continue

                    for A, B, C in itertools.product(
                        buckets[formula_A][charge_A],
                        buckets[formula_B][charge_B],
                        buckets[formula_C][charge_C],
                    ):

                        # exclude reactions already considered
                        # Since we use `combinations_with_replacement` to consider
                        # products B and C for the same formula, buckets[formula_B] and
                        # buckets[C] could be the same buckets. Then when we use
                        # itertools.product to loop over them, molecules (M, M') could
                        # be either (B, C) or (C, B), appearing twice.
                        # We can do combinations_with_replacement for the loop over
                        # charge when formula_B and formula_C are the same, but this
                        # would complicate the code. So we use reaction_ids to keep
                        # record and not include them.
                        ids = {A.id, B.id, C.id}
                        if ids in reaction_ids:
                            continue

                        bonds = is_valid_A_to_B_C_reaction(A, B, C, first_only=find_one)
                        if bonds:
                            reaction_ids.append(ids)
                            for b in bonds:
                                A2BC.append(Reaction([A], [B, C], b))

        self.reactions = A2BC

        logger.info("{} A -> B + C style reactions extracted".format(len(A2BC)))

        return A2BC

    def extract_one_bond_break(self, find_one=True):
        """
        Extract all reactions that only has one bond break or the type ``A -> B + C``
        (break a bond not in a ring) or ``A -> D`` (break a bond in a ring)

        Returns:
            A list of reactions.
        """
        A2B = self.extract_A_to_B_style_reaction(find_one)
        A2BC = self.extract_A_to_B_C_style_reaction(find_one)
        self.reactions = A2B + A2BC

        return A2B, A2BC

    @staticmethod
    def _get_formula_composition_map(mols):
        fcmap = dict()
        for m in mols:
            fcmap[m.formula] = m.composition_dict
        return fcmap

    @staticmethod
    def _is_even_composition(composition):
        for _, amt in composition.items():
            if int(amt) % 2 != 0:
                return False
        return True

    @staticmethod
    def _is_valid_A_to_B_C_composition(composition1, composition2, composition3):
        combined23 = defaultdict(int)
        for k, v in composition2.items():
            combined23[k] += v
        for k, v in composition3.items():
            combined23[k] += v
        return composition1 == combined23

    @staticmethod
    def _is_valid_A_to_B_C_charge(charge1, charge2, charge3):
        return charge1 == charge2 + charge3

    def to_file(self, filename="rxns.pkl"):
        logger.info("Start writing reactions to file: {}".format(filename))

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactions": self.reactions,
        }
        pickle_dump(d, filename)


class ReactionExtractorFromReactant:
    """
    Create reactions from reactant only, and products molecules will be created.
    
    This needs metadata indicating the bond to break in a reactant, otherwise all bonds
    in a molecule are to break.

    Args:
        molecules (list): a sequence of :class:`MoleculeWrapper`.
        bond_energies (list of dict, optional): bond energies. Each dict for one
            molecule, with bond index (a tuple) as key and bond energy as value.
        allowed_charge (list): allowed charges for the fragments. The charges of the
            products can only take values from this list and also the sum of the
            product charges has to equal the charge of the reactant. For example,
            if the reactant has charge 0 and allowed_charge is [-1, 0, 1], then the
            two products can take charges (-1, 1), (0,0) and (1, -1).
            Set to [0] if None.
    """

    def __init__(self, molecules, bond_energies=None, allowed_charge=None):
        if bond_energies is not None and allowed_charge is not None:
            if len(allowed_charge) != 1:
                raise ValueError(
                    f"expect the size of allowed_charge to be 1 when bond_energies "
                    f"is not None, but got {len(allowed_charge)} "
                )

        self.molecules = molecules
        self.bond_energies = bond_energies
        self.allowed_charge = [0] if allowed_charge is None else allowed_charge
        self.reactions = None

    def extract_with_energies(self):
        """
        Extract reactions for bonds having energy.

        Return:
            list: a sequence of :class:`Reaction`.
        """

        if self.bond_energies is None:
            raise RuntimeError(
                "`bond_energies` not provided at instantiation. Either provide it or "
                "call `extract_ignoring_energies` instead if you are doing inference."
            )
        return self._extract(self.bond_energies)

    def extract_ignoring_energies(self):
        """
        Create reactions by breaking all bonds in molecules and set the energies to None.

        Return:
            list: a sequence of :class:`Reaction`.
        """
        bond_energies = [{b: None for b, _ in m.bonds.items()} for m in self.molecules]
        return self._extract(bond_energies)

    def _extract(self, bond_energies):

        mol_reservoir = set(self.molecules)

        reactions = []
        for reactant, b_e in zip(self.molecules, bond_energies):
            for b, e in b_e.items():
                b = tuple(sorted(b))
                num_products = len(reactant.fragments[b])
                product_charges = factor_integer(
                    reactant.charge, self.allowed_charge, num_products
                )
                rxns, mols = create_reactions_from_reactant(
                    reactant, b, product_charges, e, mol_reservoir,
                )
                reactions.extend(rxns)
                mol_reservoir.update(mols)

        self.reactions = reactions

        return reactions


def create_reactions_from_reactant(
    reactant,
    broken_bond,
    product_charges,
    bond_energy=None,
    mol_reservoir=None,
    return_error=False,
):
    """
    Create reactions from reactant by breaking a bond.

    Args:
        reactant (MoleculeWrapper): reactant molecule
        broken_bond (tuple): a 2-tuple indicating the bond to break to generate products
        product_charges (list of list): each inner list gives the charge(s) of the
            product(s). Inner list could be of size 1 (one product, e.g. ring opening
            reaction) or 2 (two products).
        bond_energy (float): energy of the bond. This is allowed only when the size of
            product_charges is 1 and the products are of the same charge (e.g. [[0]] or
            [[1]]: one product and  [[0,0]] or [[1,1]] two products of the same charge.
        mol_reservoir (set): For newly created reactions, a product is first searched
            in the mol_reservoir. If existing (w.r.t. charge and isomorphism),
            the mol from the reservoir is used as the product. If not, a new mol is
            created.
        return_error (bool): If True and the creation fails, return error message
            instead of empty list.

    Returns:
        reactions (list): a sequence of Reaction, the number of reactions is
            equal to the size of product_charges. If the reactant cannot be fragmented,
            returns empty list.
        molecules (list): a sequence of MoleculeWrapper, created as the products
    """

    if bond_energy is not None:
        if len(product_charges) != 1:
            raise ValueError(
                f"expect the size of product_charges to be 1 when bond_energy is "
                f"provided, but got {len(product_charges)}"
            )
        else:
            if len(set(product_charges[0])) != 1:
                raise ValueError(
                    f"expect values of product_charges to be the same, "
                    f"but got{product_charges}"
                )

    #
    # create fragments using rdkit and then convert the rdkit fragment to wrapper mol
    #

    fragments, failed = fragment_rdkit_mol(reactant.rdkit_mol, broken_bond)

    # cannot fragment the reactant, return empty list
    if fragments is None:
        logger.error(
            f"Cannot fragment molecule `{reactant.id}` at bond {broken_bond} "
            f"because {failed}"
        )
        if return_error:
            return None, failed
        else:
            return [], []

    nf = len(fragments)
    nc = np.asarray(product_charges).shape[1]
    assert nf == nc, f"number of fragments ({nf}) not equal to number of charges ({nc})"

    # create reactions
    mol_reservoir = set(mol_reservoir) if mol_reservoir is not None else None
    reactions = []
    molecules = []
    for charges in product_charges:

        # create product molecules
        products = []
        for i, c in enumerate(charges):
            # mid needs to be unique
            mid = f"{reactant.id}_{broken_bond[0]}-{broken_bond[1]}_{c}_{i}"
            mol = rdkit_mol_to_wrapper_mol(fragments[i], charge=c, identifier=mid)

            if mol_reservoir is None:
                molecules.append(mol)
            else:
                existing_mol = search_mol_reservoir(mol, mol_reservoir)

                # not in reservoir
                if existing_mol is None:
                    molecules.append(mol)
                    mol_reservoir.add(mol)
                # in reservoir
                else:
                    mol = existing_mol

            products.append(mol)

        str_charges = "-".join([str(c) for c in charges])
        rid = f"{reactant.id}_{broken_bond[0]}-{broken_bond[1]}_{str_charges}"
        rxn = Reaction(
            [reactant],
            products,
            broken_bond=broken_bond,
            free_energy=bond_energy,
            identifier=rid,
        )
        reactions.append(rxn)

    return reactions, molecules

    #
    # create fragments by creating the wrapper mol directly from atoms and bonds
    # as a result, the rdkit_mol attribute of wrapper mol will be created internally
    #

    # fragments = reactant.fragments[broken_bond]
    #
    # nf = len(fragments)
    # nc = np.asarray(product_charges).shape[1]
    # assert nf == nc, f"number of fragments ({nf}) not equal to number of charges ({nc})"
    #
    # # fragments species, coords, and bonds (the same for products of different charges)
    # species = []
    # coords = []
    # bonds = []
    # for fg in fragments:
    #     nodes = fg.graph.nodes.data()
    #     nodes = sorted(nodes, key=lambda pair: pair[0])
    #     species.append([v["specie"] for k, v in nodes])
    #     coords.append([v["coords"] for k, v in nodes])
    #     edges = fg.graph.edges.data()
    #     bonds.append([(i, j) for i, j, v in edges])
    #
    # # create reactions
    # mol_reservoir = set(mol_reservoir) if mol_reservoir is not None else None
    # reactions = []
    # molecules = []
    # for charges in product_charges:
    #
    #     # create product molecules
    #     products = []
    #     for i, c in enumerate(charges):
    #         # mid needs to be unique
    #         mid = f"{reactant.id}_{broken_bond[0]}-{broken_bond[1]}_{c}_{i}"
    #         mol = create_wrapper_mol_from_atoms_and_bonds(
    #             species[i], coords[i], bonds[i], charge=c, identifier=mid
    #         )
    #
    #         if mol_reservoir is None:
    #             molecules.append(mol)
    #         else:
    #             existing_mol = search_mol_reservoir(mol, mol_reservoir)
    #
    #             # not in reservoir
    #             if existing_mol is None:
    #                 molecules.append(mol)
    #                 mol_reservoir.add(mol)
    #             # in reservoir
    #             else:
    #                 mol = existing_mol
    #
    #         products.append(mol)
    #
    #     str_charges = "-".join([str(c) for c in charges])
    #     rid = f"{reactant.id}_{broken_bond[0]}-{broken_bond[1]}_{str_charges}"
    #     rxn = Reaction(
    #         [reactant],
    #         products,
    #         broken_bond=broken_bond,
    #         free_energy=bond_energy,
    #         identifier=rid,
    #     )
    #     reactions.append(rxn)
    #
    # return reactions, molecules


def search_mol_reservoir(mol, reservoir):
    """
    Determine whether a mol is in a set of molecules by charge and isomorphism.

    Args:
        mol (MoleculeWrapper): the molecule to determine.
        reservoir (set): MoleculeWrapper reservoir set
        
    Returns:
        None: if not in
        m: if yes, the molecule in the reservoir

    """
    for m in reservoir:
        if m.charge == mol.charge and m.mol_graph.isomorphic_to(mol.mol_graph):
            return m
    return None


def factor_integer(x, allowed, num=2):
    """
    Factor an integer to the sum of multiple integers.

    Args:
        x (int): the integer to be factored
        allowed (list of int): allowed values for factoring.
        num (int): number of integers to sum

    Returns:
        list: factor values

    Example:
        >>> factor_integer(0, [1,0, -1], 2)
        >>> [(-1, 1), (0, 0), (1, -1)]
    """
    if num == 1:
        return [(x,)]

    elif num == 2:
        res = []
        for i, j in itertools.product(allowed, repeat=2):
            if i + j == x:
                res.append((i, j))

        return res
    else:
        raise Exception(f"not implemented for num={num} case.")


def is_valid_A_to_B_reaction(reactant, product, first_only=True):
    """
    Check whether the reactant and product can form A -> B style reaction w.r.t.
    isomorphism.

    Args:
        reactant, product (MoleculeWrapper): molecules
        first_only (bool): If `True`, only return the first found one. If `False`
            return all.

    Returns:
        list: bonds of reactant (represented by a tuple of the two atoms associated
            with the bond) by breaking which A -> B reaction is valid.
            Could be empty if no such reaction can form.
    """
    bonds = []
    for b, mgs in reactant.fragments.items():
        if len(mgs) == 1 and mgs[0].isomorphic_to(product.mol_graph):
            bonds.append(b)
            if first_only:
                return bonds
    return bonds


def is_valid_A_to_B_C_reaction(reactant, product1, product2, first_only=True):
    """
    Check whether the reactant and product can form A -> B + C style reaction w.r.t.
    isomorphism.

    Args:
        reactant, product1, product2 (MoleculeWrapper): molecules
        first_only (bool): If `True`, only return the first found one. If `False`
            return all.

    Returns:
        list: bonds of reactant (represented by a tuple of the two atoms associated
            with the bond) by breaking which A -> B + C reaction is valid.
            Could be empty if no such reaction can form.
    """

    bonds = []
    for b, mgs in reactant.fragments.items():
        if len(mgs) == 2:
            if (
                mgs[0].isomorphic_to(product1.mol_graph)
                and mgs[1].isomorphic_to(product2.mol_graph)
            ) or (
                mgs[0].isomorphic_to(product2.mol_graph)
                and mgs[1].isomorphic_to(product1.mol_graph)
            ):
                bonds.append(b)
                if first_only:
                    return bonds
    return bonds


def nx_graph_atom_mapping(g1, g2):
    """
    Mapping the atoms from g1 to g2 based on isomorphism.

    Args:
        g1, g2: nx graph

    Returns:
        dict: atom mapping from g1 to g2, but `None` is g1 is not isomorphic to g2.

    See Also:
        https://networkx.github.io/documentation/stable/reference/algorithms/isomorphism.vf2.html
    """
    nm = iso.categorical_node_match("specie", "ERROR")
    GM = iso.GraphMatcher(g1.to_undirected(), g2.to_undirected(), node_match=nm)
    if GM.is_isomorphic():
        return GM.mapping
    else:
        return None
