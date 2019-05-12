"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function


from itertools import count
from random import choice, random, shuffle, randint, gauss
import copy

import sys

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle
from neat.six_util import iteritems, iterkeys

import numpy as np

class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected')]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1','yes','true','on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0','no','false','off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write('initial_connection      = {0} {1}\n'.format(self.initial_connection,
                                                                 self.connection_fraction))
        else:
            f.write('initial_connection      = {0}\n'.format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if not 'initial_connection' in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

############# BEGIN LURYE/VIEGO CODE #############
class LayeredGenomeConfig(DefaultGenomeConfig):

    def __init__(self, params):
        super().__init__(params)
        _params = [ConfigParameter('num_hidden_per_layer', list),
                   ConfigParameter('connectivity', float)]
        for p in _params:
            setattr(self, p.name, p.interpret(params))
        self.num_hidden_per_layer = [int(i) for i in self.num_hidden_per_layer]
        self.num_layers = len(self.num_hidden_per_layer) + 2
############# END LURYE/VIEGO CODE #############

class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # Add connections based on initial connectivity type.

        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep='\n', file=sys.stderr);
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr);
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = partial_nodirect {0};".format(
                            config.connection_fraction),
                        "\tif not, set initial_connection = partial_direct {0}".format(
                            config.connection_fraction),
                        sep='\n', file=sys.stderr);
                self.connect_partial_nodirect(config)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def mutate(self, config):
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1,(config.node_add_prob + config.node_delete_prob +
                         config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob/div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob)/div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob)/div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob)/div):
                self.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        new_node_id = config.get_new_node_key(self.nodes)

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

    def add_connection(self, config, input_key, output_key, weight, enabled):
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in iterkeys(self.nodes) if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in iteritems(self.connections):
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in iterkeys(self.nodes) if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.output_keys]
        output = [i for i in iterkeys(self.nodes) if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections


    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

############# BEGIN LURYE/VIEGO CODE #############
# Note that a lot of these methods are basically copied from the DefaultGenome
# class but with some minor changes
class LayeredGenome(DefaultGenome):

    def __init__(self, key):
        super().__init__(key)

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return LayeredGenomeConfig(param_dict)

    def configure_new(self, config):
        self.layers = [[] for _ in range(config.num_layers)]

        for node_key in config.input_keys:
            self.layers[0].append(node_key)

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)
            self.layers[-1].append(node_key)

        for i in range(1, config.num_layers - 1):
            for _ in range(config.num_hidden_per_layer[i - 1]):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node
                self.layers[i].append(node_key)

        connections = self.compute_full_connections(config)
        shuffle(connections)
        num_to_add = int(round(len(connections) * config.connectivity))
        for input_id, output_id in connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config):
        output = [i for i in iterkeys(self.nodes) if i in config.output_keys]
        connections = []
        for input_id in config.input_keys:
            for h in self.layers[1]:
                connections.append((input_id, h))
        for i in range(1, len(self.layers) - 1):
            for input_id in self.layers[i]:
                for output_id in self.layers[i + 1]:
                    connections.append((input_id, output_id))
        return connections

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

        self.layers = copy.deepcopy(parent1.layers)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        if len(self.nodes) >= 30:
            return

        new_node_layer = randint(1, len(self.layers) - 2)
        new_node_id = config.get_new_node_key(self.nodes)
        self.layers[new_node_layer].append(new_node_id)
        self.nodes[new_node_id] = self.create_node(config, new_node_id)

        input_id = choice(self.layers[new_node_layer - 1])
        output_id = choice(self.layers[new_node_layer + 1])
        self.add_connection(config, input_id, new_node_id, gauss(0, 0.1), True)
        self.add_connection(config, new_node_id, output_id, gauss(0, 0.1), True)

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        
        in_layer = randint(0, len(self.layers) - 2)
        out_layer = in_layer + 1

        in_node = choice(self.layers[in_layer])
        out_node = choice(self.layers[out_layer])

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        layer_perm = list(range(1, len(self.layers) - 2))
        shuffle(layer_perm)
        del_key = -1
        del_key_layer = None
        for l in layer_perm:
            if len(self.layers[l]) > 1:
                del_key = choice(self.layers[l])
                del_key_layer = l
                break

        if del_key == -1:
            return -1

        connections_to_delete = set()
        for k, v in iteritems(self.connections):
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]
        self.layers[del_key_layer].remove(del_key)

        return del_key

    def modularity(self):
        return self._directed_louvain(self._get_adj_mat())
        
    def _map_nodes(self):
        node_map = {}
        i_next = 0
        for v in self.layers[0]:
            node_map[v] = i_next
            i_next += 1
        for v in self.nodes:
            node_map[v] = i_next
            i_next += 1
        return node_map

    def _get_adj_mat(self):
        adj = np.zeros((
            len(self.nodes) + len(self.layers[0]), 
            len(self.nodes) + len(self.layers[0])
        ))
        node_map = self._map_nodes()
        for cg in self.connections.values():
            if cg.enabled:
                input_key, output_key = cg.key
                adj[node_map[input_key], node_map[output_key]] = cg.weight
        return adj

    def _directed_louvain(self, adj):
        n_nodes = adj.shape[0]
        adj = abs(adj)
        in_deg = adj.sum(axis=0)
        out_deg = adj.sum(axis=1)
        weight_sum = adj.sum()

        if weight_sum == 0:
            return 0.

        comms = {
            i: {
                "id": i,
                "members": set([i]),        # a set of node ids indicating membership 
                "out_deg_sum": out_deg[i],  # the sum of the outdegrees of every node in this community
                "in_deg_sum": in_deg[i],    # the sum of the indegrees of every node in this community
                "k_out": adj[i].copy(),     # the number of incoming edges each node has connected to nodes in this community
                "k_in": adj[:, i].copy()    # the number of outgoing edges each node has connected to nodes in this community
            } for i in range(n_nodes)
        }

        # keep track of which community each node belongs to
        node_map = {i: i for i in range(n_nodes)}

        modularity = 1. / weight_sum * (adj.diagonal().sum() \
                     - 1. / weight_sum * (out_deg * in_deg).sum())

        while True:

            old_modularity = modularity

            # Stage 1 (greedy search over partitions)
            while True:

                # flag to know if any communities changed on this pass
                changed = False

                for i in range(n_nodes):

                    neighbors_in = set(np.where(adj[:, i] > 0)[0])
                    neighbors_out = set(np.where(adj[i] > 0)[0])
                    neighbors = neighbors_in.union(neighbors_out)

                    best_delta = 0
                    best_comm = None
                    comms_tried = set([node_map[i]])

                    for j in neighbors:

                        # if we already tried to move node i into node j's community,
                        # don't bother trying again
                        if node_map[j] in comms_tried:
                            continue    
                        comms_tried.add(node_map[j])

                        # calculate the change in modularity due to adding node i
                        # to node j's community
                        new_comm = comms[node_map[j]]
                        delta_add = 1. / weight_sum * new_comm["k_in"][i] \
                                    + 1. / weight_sum * new_comm["k_out"][i] \
                                    - in_deg[i] / (weight_sum ** 2) * new_comm["out_deg_sum"] \
                                    - out_deg[i] / (weight_sum ** 2) * new_comm["in_deg_sum"] \
                                    + 1. / weight_sum * (adj[i, i] - (in_deg[i] * out_deg[i]) / weight_sum)

                        # calculate the change in modularity due to removing node i
                        # from its current community
                        old_comm = comms[node_map[i]]
                        delta_remove = -1. / weight_sum * (old_comm["k_in"][i] - adj[i, i]) \
                                       - 1. / weight_sum * (old_comm["k_out"][i] - adj[i, i]) \
                                       + in_deg[i] / (weight_sum ** 2) * (old_comm["out_deg_sum"] - out_deg[i]) \
                                       + out_deg[i] / (weight_sum ** 2) * (old_comm["in_deg_sum"] - in_deg[i]) \
                                       - 1. / weight_sum * (adj[i, i] - (in_deg[i] * out_deg[i]) / weight_sum)

                        # keep track of the overall change
                        delta = delta_add + delta_remove
                        if delta > best_delta:
                            best_delta = delta
                            best_comm = new_comm

                    # if we found a positive change in modularity, then move node i into its new
                    # community
                    if best_delta > 0:
                        modularity += best_delta
                        changed = True
                        old_comm = comms[node_map[i]]
                        best_comm["members"].add(i)
                        best_comm["out_deg_sum"] += out_deg[i]
                        best_comm["in_deg_sum"] += in_deg[i]
                        best_comm["k_out"] += adj[i]
                        best_comm["k_in"] += adj[:, i]
                        node_map[i] = best_comm["id"]
                        old_comm["members"].remove(i)
                        if len(old_comm["members"]) == 0:
                            del comms[old_comm["id"]]
                        else:
                            old_comm["out_deg_sum"] -= out_deg[i]
                            old_comm["in_deg_sum"] -= in_deg[i]
                            old_comm["k_out"] -= adj[i]
                            old_comm["k_in"] -= adj[:, i]

                # if there are no more improvements to be made, go on to stage 2
                if not changed:
                    break

            # Stage 2 (define a new graph whose nodes are the communities from stage 1)

            # if the modularity didn't increase during stage 1, return the local maximum    
            if old_modularity == modularity:
                return modularity

            # form the new graph and put each new node into its own community   
            n_new_nodes = len(comms)
            new_adj = np.zeros((n_new_nodes, n_new_nodes))
            new_comms = {}
            k = 0
            for i in comms:
                new_comms[k] = {
                    "id": k,
                    "members": set([k]),
                    "out_deg_sum": comms[i]["out_deg_sum"],
                    "in_deg_sum": comms[i]["in_deg_sum"],
                    "k_out": np.zeros(n_new_nodes),
                    "k_in": np.zeros(n_new_nodes)
                }
                k += 1
            k = 0
            for i in comms:
                l = 0
                for j in comms:
                    new_adj[k, l] = comms[i]["k_out"][list(comms[j]["members"])].sum()
                    new_comms[k]["k_out"][l] = new_adj[k, l]
                    new_comms[l]["k_in"][k] = new_adj[k, l]
                    l += 1
                k += 1
            adj = new_adj
            comms = new_comms
            in_deg = adj.sum(axis=0)
            out_deg = adj.sum(axis=1)
            n_nodes = len(comms)
            node_map = {i: i for i in range(n_nodes)}
            weight_sum = adj.sum()

            if weight_sum == 0:
                return 0.
############# END LURYE/VIEGO CODE #############        
