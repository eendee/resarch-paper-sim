export class Paper{
    id : string;
    title: string;
}

export class ParagraphString{
    sentence : string;
    score?: number;
    Id?: number;
    originalParagraphId?: number;
    isInSpotlight = false;
}

export interface FetchPapersResponse{
    dataset: Paper[];
}

export interface SinglePaperResponse{
    dataset: ParagraphString[];
}