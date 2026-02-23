"""RNA sequences for analysis.

Includes real biological RNAs and artificial sequences designed to show
interesting clustering behavior.
"""

SEQUENCES = {
    "tRNA_Phe": {
        "sequence": (
            "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAU"
            "CUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
        ),
        "description": "Yeast phenylalanine tRNA (76 nt)",
        "type": "natural",
        "n_clusters": 2,
    },
    "5S_rRNA": {
        "sequence": (
            "GCCUACGGCCAUACCACCCUGAACGCGCCCGAUCUCGUUCGAAUCCGAGU"
            "GGUUAGGGAAACAGCAGAAGCUGGAGAAUGGGCG"
        ),
        "description": "E. coli 5S rRNA fragment (84 nt)",
        "type": "natural",
        "n_clusters": 3,
    },
    "bistable": {
        "sequence": "CGCGCGAAAAACGCGCGAAAAACGCGCG",
        "description": (
            "Artificial bistable RNA with two competing CGCGCG hairpins (28 nt). "
            "CGCGCG(1-6) pairs with CGCGCG(12-17) OR CGCGCG(12-17) pairs with "
            "CGCGCG(23-28), giving two mutually exclusive conformations."
        ),
        "type": "artificial",
        "n_clusters": 2,
    },
    "hammerhead": {
        "sequence": (
            "AACCCUCGUGUCGGGAAGACGAAACUCGACCGAAACUGAAAGUCGUCC"
            "AGGCAAAUCC"
        ),
        "description": "Hammerhead ribozyme minimal motif (58 nt)",
        "type": "natural",
        "n_clusters": 3,
    },
}
