import { Sheet, Typography, Table } from "@mui/joy"
export interface ResultTableProps {
    displayMessage: string;
}
interface Row {
    confidence: string;
    label: string;
}

export const ResultTable = ({ displayMessage }: ResultTableProps) => {
    const lines = displayMessage.split('\n');
    const mostConfident = lines[0];
    const tableRows = lines.slice(1).map((line) => {
        const match = line.match(/([\d.]+)%:\s(.+)/);
        return match ? { confidence: match[1], label: match[2] } : null;
    }).filter(Boolean) as Row[];

    const getRowColor = (confidence: number) => {
        if (confidence > 80) return '#90EE90';
        if (confidence > 40) return 'yellow';
        return '#FF7F7F';
    };

    return (
        <Sheet variant="outlined" sx={{ padding: 2, maxWidth: 600, margin: 'auto' }}>
            <Typography level="h2" sx={{ marginBottom: 2 }}>
                {mostConfident}
            </Typography>
            <Table variant="soft" borderAxis="bothBetween">
                <thead>
                    <tr>
                        <th style={{ width: '50%' }}>Confidence (%)</th>
                        <th>Label</th>
                    </tr>
                </thead>
                <tbody>
                    {tableRows.map((row, index) => (
                        <tr
                            key={index}
                            style={{
                                backgroundColor: getRowColor(parseFloat(row.confidence)),
                            }}
                        >
                            <td>{row.confidence}</td>
                            <td>{row.label}</td>
                        </tr>
                    ))}
                </tbody>
            </Table>
        </Sheet>
    )
}
