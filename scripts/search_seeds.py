class search_seed:
    def __init__(self, 
                 miRNA:str, 
                 mRNA:str):
        self.miRNA = miRNA
        self.mRNA = mRNA
                
    @staticmethod
    def is_WC_pair(n1: str, n2: str):
        """
        Returns True if the two nucleotides n1 and n2 form a Watson-Crick base pair:
            - A-T (or A-U)
            - G-C
        """
        naux1 = n1.upper()
        naux2 = n2.upper()
        if (naux1 == 'A' and naux2 in ['U', 'T']):
            return True
        if (naux2 == 'A' and naux1 in ['U', 'T']):
            return True
        if (naux1 == 'G' and naux2 == 'C'):
            return True
        if (naux2 == 'G' and naux1 == 'C'):
            return True
        return False
    
    @staticmethod
    def is_Wobble_pair(n1: str, n2: str) -> bool:
        """
        Returns true if the two nucleotides form a Wobble base pair:
         - G-U
         - G-T
         - I-A, I-C, I-U (if one of them is 'I' and the other is not 'G')
        """
        naux1 = n1.upper()
        naux2 = n2.upper()

        # If one is 'I' (Inosine), then check it pairs with A/C/U but not G
        if naux1 == 'I' or naux2 == 'I':
            # If one is I, the other is A, C, or U
            # The original condition from Java: (naux1 != 'G' && naux2 != 'G')
            return (naux1 != 'G' and naux2 != 'G')

        # G-U / G-T
        return ((naux1 == 'G' and naux2 in ['T', 'U']) or
                (naux2 == 'G' and naux1 in ['T', 'U']))

    def find_seed_match(self, 
                        min_seed_length = 6):
        """
        Find seed match between miRNA sequence and mRNA site.
        
        Allow 4 cases:
            1. 8-mer (perfect 8-nt match)
            2. 7-mer-m8 (match at position 2-8 of miRNA)
            3. 7-mer-t1 (match at position 1-7 of miRNA)
            4. 6-mer (match at position 2-7 of miRNA)
        
        Args:
            miRNA (str): miRNA sequence (5' -> 3')
            mRNA (str): 30-nt mRNA site sequence (3' -> 5')
        
        Returns:
            list of tuples (miRNA_end, mRNA_end, match_length)
        """
        m_len = len(self.miRNA)
        r_len = len(self.mRNA)

        # DP table for storing lengths of contiguous matches
        dp = [[0] * (r_len + 1) for _ in range(m_len + 1)]

        matches = []

        for i in range(1, m_len + 1):
            for j in range(1, r_len + 1):
                mirna_base = self.miRNA[i - 1]
                mrna_base = self.mRNA[j - 1]

                # Check if nucleotides form Watson-Crick pair
                if search_seed.is_WC_pair(mirna_base, mrna_base) or search_seed.is_Wobble_pair(mirna_base, mrna_base):
                    dp[i][j] = dp[i - 1][j - 1] + 1

                    # Check if contiguous match meets the minimum seed length
                    if dp[i][j] >= min_seed_length:
                        match_length = dp[i][j]
                        matches.append((i - 1, j - 1, match_length))
                else:
                    dp[i][j] = 0

        return matches

    def find_seed_matches(self, 
                          rev=False,
                          min_seed_length=6):
        """
        Find all seed matches in an mRNA sequence.

        Args:
            miRNA (str): miRNA sequence (5' -> 3')
            mRNA (str): mRNA sequence (3' -> 5')
            rev (bool): If True, reverse sequences before searching

        Returns:
            List of tuples containing the pretty-printed miRNA/mRNA alignment
        """
        if rev:
            self.mRNA = self.mRNA[::-1]
            self.miRNA = self.miRNA[::-1]

        matches = self.find_seed_match(min_seed_length=min_seed_length)
        matches_pretty_print = []

        for mirna_end, mrna_end, length in matches:
            mirna_start = mirna_end - length + 1
            mrna_start = mrna_end - length + 1

            # Create visualization of alignment
            result_binding = list('.' * len(self.mRNA))
            for index in range(mrna_start, mrna_end + 1):
                result_binding[index] = 'l'
            result_binding = ''.join(result_binding)

            # Adjust miRNA sequence to match mRNA positions
            start_diff = mrna_start - mirna_start
            miRNA_pretty_print = list('.' * start_diff) + list(self.miRNA) + list('.' * (len(self.mRNA) - len(self.miRNA) - start_diff))
            miRNA_pretty_print = ''.join(miRNA_pretty_print)

            matches_pretty_print.append((miRNA_pretty_print, self.mRNA, result_binding))

        return matches_pretty_print

if __name__ == '__main__':
    miRNA = "UAAAGUGCUUAUAGUGCAGGUAG"
    mRNA = "CTATTAAAGAGAACAAATTTTTACTTAAGCACTTTGAGGT"
    search_seed_ins = search_seed(miRNA, mRNA)
    matches = search_seed_ins.find_seed_matches(rev=True)
    for miRNA, mRNA, binding in matches: 
        print(miRNA)
        print(mRNA)
        print(binding)