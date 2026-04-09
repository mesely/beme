import copy


def evolve_sector(funds, sector_name):
    """
    Run one evolutionary step for the given sector.
    Mutates the funds list in-place.
    """
    sector_funds = [f for f in funds if f['sector'] == sector_name]
    sector_funds.sort(key=lambda x: x['balance'], reverse=True)

    parent = sector_funds[0]
    child  = sector_funds[-1]

    child['model']     = copy.deepcopy(parent['model'])
    transfer           = parent['balance'] * 0.1
    parent['balance'] -= transfer
    child['balance']   = max(100.0, transfer)


def evolve(funds):
    """Run one evolutionary step for all sectors."""
    for sector_name in ["COMMODITY", "FINANCE"]:
        evolve_sector(funds, sector_name)
