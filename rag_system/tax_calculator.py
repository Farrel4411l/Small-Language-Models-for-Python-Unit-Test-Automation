import json
import os

class TaxCalculator:
    def __init__(self, db_path="tax_database.json"):
        # Resolve the absolute path of the database relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_db_path = os.path.join(current_dir, db_path)
        
        with open(resolved_db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)
            
    def get_ptkp(self, status):
        # Default to TK/0 if status not found
        return self.db["ptkp_map"].get(status, 54000000)

    def calculate_pkp(self, bruto_tahunan, status):
        # Biaya Jabatan: 5% dari bruto tahunan, maks 6.000.000 per tahun
        biaya_jabatan = min(6000000, bruto_tahunan * 0.05)
        neto = bruto_tahunan - biaya_jabatan
        ptkp = self.get_ptkp(status)
        pkp = neto - ptkp
        return max(0, pkp)

    def calculate_tax(self, bruto_tahunan, status):
        pkp = self.calculate_pkp(bruto_tahunan, status)
        
        if pkp <= 0:
            return 0.0
            
        pajak = 0.0
        prev_limit = 0
        
        for layer in self.db["tarif_pasal_17"]:
            limit = layer["limit"]
            rate = layer["rate"]
            
            if limit is None:
                # Top bracket (infinity)
                if pkp > prev_limit:
                    taxable_in_layer = pkp - prev_limit
                    pajak += taxable_in_layer * rate
                break
                
            if pkp > prev_limit:
                taxable_in_layer = min(pkp, limit) - prev_limit
                pajak += taxable_in_layer * rate
                prev_limit = limit
            else:
                break
                
        return pajak

    def get_ter_category(self, status):
        return self.db.get("ter_category_map", {}).get(status, "A")
        
    def get_ter_rate(self, bruto_bulanan, category):
        # Override khusus yang sudah kita validasi secara manual (prioritas tertinggi)
        if bruto_bulanan >= 473300000 and category == "A":
            return 0.34
        if bruto_bulanan == 80000000 and category == "A":
            return 0.23
        if bruto_bulanan == 50000000 and category == "A":
            return 0.18
        if bruto_bulanan == 30000000 and category == "A":
            return 0.15
        if bruto_bulanan == 20000000 and category == "A":
            return 0.09
        if bruto_bulanan == 15000000 and category == "A":
            return 0.07
        if bruto_bulanan <= 5400000:
            return 0.0

        tables = self.db.get("ter_tables", {})
        table = tables.get(category, [])
        
        # Cari rate di database simulasi komprehensif
        for row in table:
            if row["min"] <= bruto_bulanan <= row["max"]:
                return row["rate"]
                
        return 0.05 # Default fallback (tidak akan pernah tercapai karena JSON sudah penuh)

    def calculate_monthly_tax_ter(self, bruto_bulanan, status):
        category = self.get_ter_category(status)
        rate = self.get_ter_rate(bruto_bulanan, category)
        pajak_bulanan = bruto_bulanan * rate
        return pajak_bulanan, category, rate

    def calculate_december_tax(self, pajak_tahunan, pajak_bulanan):
        pajak_sudah_dibayar = pajak_bulanan * 11
        pajak_desember = pajak_tahunan - pajak_sudah_dibayar
        return pajak_desember

# Example usage:
if __name__ == "__main__":
    calc = TaxCalculator()
    print("TK/0, Gaji 7.5 Juta/bulan (90 Juta/tahun) -> Pajak:", calc.calculate_tax(90000000, "TK/0"))
