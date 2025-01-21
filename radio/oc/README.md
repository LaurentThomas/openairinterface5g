[[_TOC_]]

# Open Cells SDR device documentation

## General

OAI works with open cells SDR, compatible with USRP config files

Example files can be found in the `ci-scripts/conf_files/` directory with a
`usrp` in the name, for instance
[`gnb.sa.band78.106prb.usrpn310.ddsuu-2x2.conf`](../../ci-scripts/conf_files/gnb.sa.band78.106prb.usrpn310.ddsuu-2x2.conf).

## Configuration

to use OC SDR driver, add --device.name oai_ocdevif on the command line, or the equivalent value in the configuration file (any OAI executable)

You can specify to use external or interal clock or time source either by
adding the parameters in the `sdr_addrs` field or by using the fields
`clock_src` or `time_src`

Valid choices for clock and time source are `internal`, `external`, and `gpsdo`.

```bash
device = {
name="oai_ocdevif";
}

RUs = (
{
  local_rf       = "yes"
  nb_tx          = 2
  nb_rx          = 2
  att_tx         = 0
  att_rx         = 0;
  bands          = [78];
  max_pdschReferenceSignalPower = -27;
  max_rxgain                    = 75;
  eNB_instances  = [0];
}
);
```

