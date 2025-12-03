import os
from collections import Counter
from dataclasses import dataclass, field

@dataclass
class PreToken:
    # unique id so that class is hashable for quick retrieval
    id: int  
    # pretoken represented as a sequence of bytes according to current vocab
    bytestring: tuple[bytes]
    # number of occurrence of this pretoken in the corpus
    num_occurrence: int
    # mapping of bp -> num occurrence of bp in this pretoken
    bp_count: dict[tuple[bytes, bytes], int] = field(init=False)

    def __post_init__(self):
        self._update_bp_count()

    def _update_bp_count(self):
        self.bp_count = Counter(zip(self.bytestring[:-1], self.bytestring[1:]))

    # updates bp_count and bytestring and returns the old bp_count. Note: bp is short for byte-pair
    def merge_and_update(self, bp:tuple[bytes, bytes]) -> dict[tuple[bytes, bytes],int]:
        new_bytestring = []
        i = 0
        while i<len(self.bytestring):
            if self.bytestring[i:i+2]==bp:
                new_bytestring.append(bp[0]+bp[1])
                i += 2
            else:
                new_bytestring.append(self.bytestring[i])
                i += 1
        self.bytestring = tuple(new_bytestring)
        old_bp_count = self.bp_count
        self._update_bp_count()
        return old_bp_count
    
    def __hash__(self) -> int:
        return self.id
    
    def __eq__(self, other: object, /) -> bool:
        return isinstance(other, PreToken) and other.id==self.id
            

@dataclass
class BytePair:
    bp: tuple[bytes, bytes]
    # list of pretokens that contain this bp
    parents: set[PreToken] = field(default_factory=lambda:set(), init=False)
    # count of bp in the entire corpus
    count: int = field(default=0, init=False)

    def add_parent(self, parent:PreToken):
        self.parents.add(parent)
        self.count += parent.bp_count[self.bp]*parent.num_occurrence
    
    def remove_parent(self, parent: PreToken, old_bp_count: dict[tuple[bytes, bytes],int]):
        self.parents.remove(parent)
        self.count -= (parent.num_occurrence*old_bp_count[self.bp])


def train(
    pretokens: Counter[str], # obtained after pretokenization step
    vocab_size:int,
    special_tokens:list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size>(min:=256+len(special_tokens)), f"Vocab size must be at least {min}"

    vocab: dict[int, bytes] = {i:tok.encode("utf-8") for i,tok in enumerate(special_tokens)}
    vocab.update({len(special_tokens)+i : bytes([i]) for i in range(256)})
    merges: list[tuple[bytes, bytes]] = []
    bps:dict[tuple[bytes, bytes], BytePair] = {}
    for i, (pretoken, count) in enumerate(pretokens.items()):
        pretok = PreToken(i, tuple(bytes([tok]) for tok in pretoken.encode("utf-8")), count)
        for bp in pretok.bp_count.keys():
            if bp not in bps:
                bps[bp] = BytePair(bp)
            bps[bp].add_parent(pretok)
    
    while len(vocab) < vocab_size :
    
        # Find bytepair(s) with maximum count
        max_count:int = 0
        maxes:list[tuple[bytes,bytes]] = []
        for bp in bps.values():
            if bp.count > max_count:
                max_count = bp.count
                maxes = [bp.bp]
            elif bp.count == max_count:
                maxes.append(bp.bp)
            # print(bp.bp, bp.count)

        # print(max_count, maxes) 
        
        # Find lexicographically greatest pair among tiebreakers
        bp_to_merge = max(maxes)

        # Update all pretokens that are affected by the merge (i.e., contain the merged bytepair)
        for pretok in bps[bp_to_merge].parents:
            # e.g., Pretok: l,o,w,e,s,t, merge "s,t"
            # old_bp_count: {(l,o):1, (o,w):1, (w,e):1, (e,s):1, (s,t):1}
            # new_bp_count: {(l,o):1, (o,w):1, (w,e):1, (e,st): 1 }
            # bps_to_remove = {(e,s), (s,t)}
            # bps_to_add = {(e,st)}
            old_bp_count = pretok.merge_and_update(bp_to_merge)
            new_bp_count = pretok.bp_count

            # find byte pairs that no longer exist due to the merge
            bps_to_remove = old_bp_count.keys() - new_bp_count.keys()

            for bp in bps_to_remove:
                # don't worry about the byte pair to merge this iteration -- the entire BytePair obj will be deleted anyways!
                if bp==bp_to_merge:
                    continue
                # remove PreToken parent from this bytepair
                bps[bp].remove_parent(pretok, old_bp_count)

            # find new byte pairs created due to the merge
            bps_to_add = new_bp_count.keys() - old_bp_count.keys()
            # print(f"pretok: {pretok.bytestring}, bps_to_remove: {bps_to_remove}, bps_to_add: {bps_to_add}")
            for bp in bps_to_add:
                if bp not in bps:
                    bps[bp] = BytePair(bp)
                bps[bp].add_parent(pretok)

        # no longer needed since the bytepair (i.e., 2 tokens) is now considered a single token              
        del bps[bp_to_merge]

        vocab[len(vocab)] = bp_to_merge[0]+bp_to_merge[1]
        merges.append(bp_to_merge)
    return vocab, merges