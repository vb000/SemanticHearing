import json


class Ontology(object):
    ANIMAL = '/m/0jbk'
    SOUNDS_OF_THINGS = '/t/dd00041'
    HUMAN_SOUNDS = '/m/0dgw9r'
    NATURAL_SOUNDS = "/m/059j3w"
    SOURCE_AMBIGUOUS_SOUNDS = "/t/dd00098"
    CHANNEL_ENVIRONMENT_BACKGROUND = "/t/dd00123"
    MUSIC = "/m/04rlf"
    ROOT = "__root__"
    
    def __init__(self, path) -> None:
        with open(path, 'rb') as f:
            ontology_list = json.load(f)

        root_node = {}
        root_node['child_ids'] = [self.SOURCE_AMBIGUOUS_SOUNDS,
                                self.ANIMAL,  
                                self.SOUNDS_OF_THINGS,
                                self.HUMAN_SOUNDS,
                                self.NATURAL_SOUNDS,
                                self.CHANNEL_ENVIRONMENT_BACKGROUND,
                                self.MUSIC]
        root_node['id'] = self.ROOT
        root_node['name'] = self.ROOT
        ontology_list.append(root_node)

        self.ontology = {item['id']: item for item in ontology_list}

        self._dfs()
        self.mark_source_ambiguous_sounds()
    
    def _dfs(self, node_id=None):
        if node_id is None:
            node_id = self.ROOT
            self.ontology[node_id]['depth'] = 0
            self.ontology[node_id]['parent_id'] = None
        else:
            parent_node = self.ontology[node_id]['parent_id']
            self.ontology[node_id]['depth'] = self.ontology[parent_node]['depth'] + 1
        
        self.ontology[node_id]['source_ambiguous'] = 0
        
        for child_id in self.ontology[node_id]['child_ids']:
            self.ontology[child_id]['parent_id'] = node_id
            self._dfs(node_id=child_id)
        
    def mark_source_ambiguous_sounds(self, node_id=None):
        if node_id is None:
            node_id = self.SOURCE_AMBIGUOUS_SOUNDS
        
        self.ontology[node_id]['source_ambiguous'] = 1   
        
        for child_id in self.ontology[node_id]['child_ids']:
            self.mark_source_ambiguous_sounds(child_id)

    def is_source_ambiguous(self, node_id):
        # print("NODE", node_id)
        return self.ontology[node_id]['source_ambiguous']
    
    def get_label(self, node_id):
        return self.ontology[node_id]['name']

    def get_id_from_name(self, name):
        for _id in self.ontology:
            if self.ontology[_id]['name'] == name:
                return _id
        
        assert 0, f"Could not find AudioSet class with name \'{name}\'"

    def unsmear(self, args):
        x = sorted(args, key = lambda x: -self.ontology[x]['depth'])

        unsmeared = []
        removed = []
        for i in range(len(args)):
            if i in removed:
                continue
            node_id = args[i]
            unsmeared.append(node_id)
            while self.ontology[node_id]['parent_id'] is not None:
                node_id = self.ontology[node_id]['parent_id']
                try:
                    idx = args.index(node_id)
                    removed.append(idx)
                except:
                    pass

        return unsmeared


    def is_leaf_node(self, id):
        assert id in self.ontology.keys(), "id not in ontology"

        if self.ontology[id]['child_ids'] == []:
            return True
        
        return False

    def get_ancestor_ids(self, id):
        assert id in self.ontology.keys(), "id not in ontology"

        ancestor_ids = [id]
        parent_id = self.ontology[id]['parent_id']
        while parent_id is not None:
            ancestor_ids.append(parent_id)
            parent_id = self.ontology[parent_id]['parent_id']
        return list(reversed(ancestor_ids))

    def is_reachable(self, parent, child):
        assert parent in self.ontology.keys(), "parent not in ontology"
        assert child in self.ontology.keys(), "child not in ontology"

        if parent == child:
            return True
        for child_id in self.ontology[parent]['child_ids']:
            if self.is_reachable(child_id, child):
                return True
        return False

    def get_leaf_nodes(self, ids):
        leaf_nodes = []
        for id in ids:
            if self.is_leaf_node(id):
                leaf_nodes.append(id)
        
        return leaf_nodes

    def get_unique_leaf_node(self, ids):
        leaf_nodes = self.get_leaf_nodes(ids)

        if len(leaf_nodes) != 1:
            return None

        return leaf_nodes[0]

    def is_unique_branch(self, ids, debug=False):
        ids = sorted(ids, key=lambda x: -self.ontology[x]['depth'])
        
        bottom = ids[0]

        ancestor_ids = self.get_ancestor_ids(bottom)
        
        for _id in ids:
            if _id not in ancestor_ids:
                return False
            
        return True